# RLT bridge claims

A claim JSON is created here before any queued GPU job begins. If a claim exists without a terminal result, the worker refuses to rerun that job automatically; the attempt is treated as infrastructure-ambiguous and requires a fresh job ID.
