from gymnasium.envs.registration import register
from .chatshop_env import IPOTextEnv


register(
     id="IPOEnv-v0",
     entry_point="web_agent_site.envs:IPOTextEnv",
)