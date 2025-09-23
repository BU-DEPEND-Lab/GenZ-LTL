from .curriculum import ZONES_CURRICULUM, ZONES_SAFETY_CURRICULUM, LETTER_CURRICULUM, LETTER_SAFETY_CURRICULUM, FLATWORLD_CURRICULUM
from .curriculum_sampler import CurriculumSampler

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'PointLtlSafety2-v0': ZONES_SAFETY_CURRICULUM,
    'LetterEnv-v0': LETTER_CURRICULUM,
    'LetterSafetyEnv-v0': LETTER_SAFETY_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
}
