from algos.srcpo.agent import Agent as SRCPO
from algos.sdac.agent import Agent as SDAC
from algos.cvarcpo.single.agent import Agent as CVaRCPOSingle
from algos.cvarcpo.multi.agent import Agent as CVaRCPOMulti
from algos.cppo.agent import Agent as CPPO
from algos.wcsac.agent import Agent as WCSAC
from algos.sdpo.agent import Agent as SDPO

algo_dict = {}
algo_dict['srcpo'] = SRCPO
algo_dict['sdac'] = SDAC
algo_dict['cvarcpo_single'] = CVaRCPOSingle
algo_dict['cvarcpo_multi'] = CVaRCPOMulti
algo_dict['cppo'] = CPPO
algo_dict['wcsac'] = WCSAC
algo_dict['sdpo'] = SDPO