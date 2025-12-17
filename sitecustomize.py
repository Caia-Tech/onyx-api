import os

# Keep this repo's tests isolated from globally installed pytest plugins that may
# require native deps or external services.
if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") is None:
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
