# Local application imports
from bo4mob.optimizers.saasbo import SAASBOStrategy
from bo4mob.optimizers.spsa import SPSAStrategy
from bo4mob.optimizers.turbo import TurboStrategy
from bo4mob.optimizers.vanillabo import VanillaBOStrategy

strategy_registery = {
    "spsa": SPSAStrategy,
    "vanillabo": VanillaBOStrategy,
    "saasbo": SAASBOStrategy,
    "turbo": TurboStrategy,
}
