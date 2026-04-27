from agents.fql import FQLAgent
from agents.fql_mt import FQLMTAgent
from agents.fql_v import FQLVAgent
from agents.fql_ar import FQLARAgent
from agents.fql_pi import FQLPIAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.nfql import NFQLAgent
from agents.nfql_2 import NFQL2Agent
from agents.nfql_3 import NFQL3Agent
from agents.nfql_4 import NFQL4Agent
from agents.nfql_5 import NFQL5Agent
from agents.nfql_6 import NFQL6Agent
from agents.nfql_7 import NFQL7Agent
from agents.nfql_8 import NFQL8Agent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fql=FQLAgent,
    fql_mt=FQLMTAgent,
    fql_v=FQLVAgent,
    fql_ar=FQLARAgent,
    fql_pi=FQLPIAgent,
    nfql=NFQLAgent,
    nfql_2=NFQL2Agent,
    nfql_3=NFQL3Agent,
    nfql_4=NFQL4Agent,
    nfql_5=NFQL5Agent,
    nfql_6=NFQL6Agent,
    nfql_7=NFQL7Agent,
    nfql_8=NFQL8Agent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
