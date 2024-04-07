import datetime
from .continual_learning import make_continual_rl_env
from .multi_agent_learning import make_multi_agent_env
from .offline_rl import make_offline_rl_dataset

name = "advanced-topics-in-deep-rl"
package_name = "adrl"
author = "Theresa Eimer"
author_email = "t.eimer@ai.uni-hannover.de"
description = "No description given"
url = "https://www.automl.org"
project_urls = {
    "Source Code": "https://github.com/automl-edu/advanced-topics-in-deep-rl",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, TheEimer"
version = "0.0.1"

__all__ = ["make_continual_rl_env", "make_multi_agent_env", "make_offline_rl_dataset"]
