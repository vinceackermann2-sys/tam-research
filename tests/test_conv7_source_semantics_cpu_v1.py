"""CPU-only frozen Conv7 structural regression checks; NOT a scientific benchmark.

These AST checks complement, but never replace, the SHA-pinned execution lock.
They intentionally avoid importing/instantiating the 100M-parameter model.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "architectures/cortex_s/reduced_attention_200m_panel_v2.py"
CONV7_PATH = ROOT / "architectures/cortex_s/attention8_causal_conv7_v1.py"


def same(node: ast.AST, expr: str) -> bool:
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(expr, mode="eval").body, include_attributes=False
    )


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError("Conv7 source drift: " + message)


def one_definition(body: list[ast.stmt], name: str, kind: type):
    found = [x for x in body if isinstance(x, kind) and x.name == name]
    require(len(found) == 1, f"definition of {name}")
    return found[0]


def named_assignment(body: list[ast.stmt], name: str) -> ast.AST:
    found = []
    for statement in body:
        targets = (
            statement.targets if isinstance(statement, ast.Assign)
            else [statement.target] if isinstance(statement, ast.AnnAssign)
            else []
        )
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            found.append(statement.value)
    require(len(found) == 1, f"assignment of {name}")
    return found[0]


def kw(call: ast.Call, name: str) -> ast.AST:
    vals = [arg.value for arg in call.keywords if arg.arg == name]
    require(len(vals) == 1, f"keyword {name}")
    return vals[0]


def analyze(panel_code: str, conv_code: str) -> dict[str, int]:
    panel = ast.parse(panel_code)
    conv = ast.parse(conv_code)
    for body, name, number in (
        (panel.body, "D_MODEL", 512),
        (panel.body, "N_LAYERS", 24),
        (conv.body, "KERNEL_SIZE", 7),
        (conv.body, "FF_HIDDEN", 2729),
    ):
        val = named_assignment(body, name)
        require(isinstance(val, ast.Constant) and type(val.value) is int and val.value == number,
                f"frozen {name}")

    parent = named_assignment(panel.body, "ATTENTION_8")
    require(isinstance(parent, ast.Call), "Attention-8 parent constructor")
    require(same(kw(parent, "attention_layers_one_based"),
                 "(3, 6, 9, 12, 15, 18, 21, 24)"), "attention schedule")
    require(same(kw(parent, "ff_hidden"), "2731"), "parent FFN width")
    require(same(named_assignment(conv.body, "SCHEDULE"),
                 "ATTENTION_8.attention_layers_one_based"), "parent schedule reused")

    dense = one_definition(panel.body, "DenseFeedForward", ast.ClassDef)
    dense_init = one_definition(dense.body, "__init__", ast.FunctionDef)
    linears = [x for x in ast.walk(dense_init)
               if isinstance(x, ast.Call) and same(x.func, "nn.Linear")]
    require(len(linears) == 2 and all(same(kw(x, "bias"), "False") for x in linears),
            "two bias-free feedforward projections")

    sparse = one_definition(conv.body, "CausalConv7ScheduledBlock", ast.ClassDef)
    init = one_definition(sparse.body, "__init__", ast.FunctionDef)
    forward = one_definition(sparse.body, "forward", ast.FunctionDef)
    conv_calls = [x for x in ast.walk(init)
                  if isinstance(x, ast.Call) and same(x.func, "nn.Conv1d")]
    require(len(conv_calls) == 1, "exactly one convolution")
    layer = conv_calls[0]
    require(len(layer.args) == 2 and all(same(x, "D_MODEL") for x in layer.args),
            "input/output channel count")
    for key, value in [("kernel_size", "KERNEL_SIZE"),
                       ("groups", "D_MODEL"), ("bias", "False")]:
        require(same(kw(layer, key), value), f"depthwise Conv1d {key}")
    gain = [x for x in ast.walk(init) if isinstance(x, ast.Assign) and
            any(isinstance(t, ast.Attribute) and t.attr == "conv_gain" for t in x.targets)]
    require(len(gain) == 1 and same(gain[0].value, "nn.Parameter(torch.zeros(()))"),
            "zero-initialized mixer gain")
    pads = [x for x in ast.walk(forward)
            if isinstance(x, ast.Call) and same(x.func, "F.pad")]
    require(len(pads) == 1 and len(pads[0].args) == 2 and
            same(pads[0].args[1], "(KERNEL_SIZE - 1, 0)"), "causal left-only padding")
    require(any(isinstance(x, ast.Assign) and len(x.targets) == 1 and
                isinstance(x.targets[0], ast.Name) and x.targets[0].id == "x" and
                same(x.value, "x + self.conv_gain * mixed") for x in forward.body),
            "zero-gated residual path")
    returns = [x for x in forward.body if isinstance(x, ast.Return)]
    require(len(returns) == 1 and
            same(returns[0].value, "x + self.feedforward(self.norm_ff(x))"),
            "feedforward residual")

    model = one_definition(conv.body, "Attention8CausalConv7LM", ast.ClassDef)
    model_init = one_definition(model.body, "__init__", ast.FunctionDef)
    loops = [x for x in model_init.body if isinstance(x, ast.For)]
    require(len(loops) == 1, "single block replacement loop")
    loop = loops[0]
    require(any(isinstance(x, ast.If) and same(x.test, "block.has_attention") and
                any(isinstance(y, ast.Continue) for y in x.body)
                for x in loop.body), "attention blocks not replaced")
    assignments = {ast.unparse(x) for x in loop.body if isinstance(x, ast.Assign)}
    require({"upgraded.norm_ff = block.norm_ff",
             "upgraded.feedforward = block.feedforward",
             "self.blocks[index] = upgraded"}.issubset(assignments),
            "share original FFN/normalization and replace only FFN-only blocks")

    # Parameter accounting isolates the width confound; it is not an ablation.
    width_matched = 101795328 - 24 * 2 * 512 * (2731 - 2729)
    conv7 = width_matched + 16 * (512 * 7 + 1)
    require((width_matched, conv7) == (101746176, 101803536), "parameter accounting")
    return {"attention_blocks": 8, "conv_blocks": 16,
            "width_matched_no_mixer_params": width_matched,
            "conv7_params": conv7, "extra_mixer_params": conv7 - width_matched}


class TestConv7SourceSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = PANEL_PATH.read_text(encoding="utf-8")
        cls.conv = CONV7_PATH.read_text(encoding="utf-8")

    def test_actual_architecture_and_capacity_confound(self):
        info = analyze(self.panel, self.conv)
        self.assertEqual(info["conv_blocks"], 16)
        self.assertEqual(info["extra_mixer_params"], 57360)
        self.assertEqual(info["conv7_params"], 101803536)

    def must_reject(self, *, panel=None, conv=None, error=""):
        with self.assertRaisesRegex(ValueError, error):
            analyze(panel if panel is not None else self.panel,
                    conv if conv is not None else self.conv)

    def test_right_padding_rejected(self):
        self.must_reject(conv=self.conv.replace("(KERNEL_SIZE - 1, 0)",
                                                "(0, KERNEL_SIZE - 1)"), error="causal left")

    def test_dense_instead_of_depthwise_rejected(self):
        self.must_reject(conv=self.conv.replace("groups=D_MODEL", "groups=1"), error="groups")

    def test_nonzero_gate_rejected(self):
        self.must_reject(conv=self.conv.replace("torch.zeros(())", "torch.ones(())"), error="zero-initialized")

    def test_ungated_mixer_rejected(self):
        self.must_reject(conv=self.conv.replace("self.conv_gain * mixed", "mixed"), error="zero-gated")

    def test_attention_layers_replaced_rejected(self):
        self.must_reject(conv=self.conv.replace("if block.has_attention:",
                                                "if not block.has_attention:"), error="attention blocks")

    def test_ffn_module_reinitialization_rejected(self):
        self.must_reject(conv=self.conv.replace("upgraded.feedforward = block.feedforward",
                                                "upgraded.feedforward = DenseFeedForward(2729)"), error="share original")

    def test_parent_ffn_bias_rejected(self):
        self.must_reject(panel=self.panel.replace("hidden, bias=False", "hidden, bias=True"),
                         error="bias-free")

    def test_parent_attention_schedule_rejected(self):
        self.must_reject(panel=self.panel.replace("(3, 6, 9, 12, 15, 18, 21, 24)",
                                                "(1, 6, 9, 12, 15, 18, 21, 24)"),
                         error="attention schedule")

    def test_conv_width_rejected(self):
        self.must_reject(conv=self.conv.replace("FF_HIDDEN: Final = 2_729",
                                                "FF_HIDDEN: Final = 2_731"), error="FF_HIDDEN")


if __name__ == "__main__":
    unittest.main()
