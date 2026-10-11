# Colab compute from the desktop chat

[Open the TAM Research connection notebook](https://colab.research.google.com/github/vinceackermann2-sys/tam-research/blob/main/colab/TAM_Research_Compute.ipynb).

The notebook checks out the public repository, records its exact Git commit,
installs the Python package, and runs a tiny device smoke test. It starts with a
CPU runtime and does not launch training or consume a queued experiment.
No GitHub token or Google Drive mount is needed for this connection.

## Connect the local agent

Google's official [Colab MCP server](https://github.com/googlecolab/colab-mcp)
connects a local agent to the Colab browser. Codex desktop and CLI share MCP
configuration. The agent must run on the same computer as the Colab browser.

Install the reviewed server version (requires `uv`, which installs Python 3.13
if needed):

```sh
uv tool install --python 3.13 git+https://github.com/googlecolab/colab-mcp@b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1
codex mcp add colab -- colab-mcp
```

On Windows, use the absolute path of `colab-mcp.exe` if it is not on PATH. In
the desktop app, reload the MCP configuration or reopen the chat to load the
new tools if necessary. Ask the agent to call
`open_colab_browser_connection`, then approve the Colab browser's connection
prompt. This grants the local agent notebook editing and code execution in the
connected Colab session. The server uses a temporary token and a localhost
WebSocket restricted to Colab origins; it does not publish an internet endpoint.

Open the connection notebook in the paired Colab browser. Run its setup and
smoke cells. The agent can then execute bounded repo commands in notebook cells
and read their outputs. Keep the connected tab open; reconnect after the local
server or browser session restarts. This local bridge is for the desktop app;
an ordinary ChatGPT web conversation cannot reach this computer's localhost
server directly.

## Hardware and results

After the CPU connection works, choose A100, L4, G4, or T4 in Colab's runtime
settings for CUDA work and rerun setup after the runtime resets. Availability,
idle limits, session duration, and compute-unit consumption depend on the Colab
plan and remaining units. A hardware selector does not guarantee an allocation.
This repo's existing CUDA training scripts need adaptation to use a TPU.

The smoke test writes `/content/tam-results/connection-smoke.json`. Keep new
run artifacts under `/content/tam-results` and download them before disconnecting.
Colab runtime storage is temporary. The notebook's final cell downloads a ZIP
of that folder; large checkpoints may need a separately approved storage path.
Existing checkpoints and datasets on the Modal volume do not transfer
automatically, and Modal-specific launchers are not Colab launchers.

The older `exp/rlt-colab` notebook is a separate experiment queue worker. This
connection does not invoke it. Use an explicitly chosen experiment configuration
and data source for research runs, and preserve the repo's scientific protocol.
