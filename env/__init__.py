from gymnasium import register
# Registrar for the gymnasium environment
# ... for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='env.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname',
    disable_env_checker=True, # Enable this for non-standard output in step() function gym(0.24.0)
)
