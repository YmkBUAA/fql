from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.nfql import NFQLAgent
from agents.nfql_2 import NFQL2Agent
from agents.nfql_3 import NFQL3Agent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent

agents = dict(
    fql=FQLAgent,
    nfql=NFQLAgent,
    nfql_2=NFQL2Agent,
    nfql_3=NFQL3Agent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
)
