from enum import Enum
from pydantic import BaseModel, Field


class MedicalPredicate(Enum):
    """
    These are the eight predicates that are necessary and sufficient to handle pairwise entity relationships.
    """
    REQUIRES_DIAGNOSTIC_TEST = "Pre-treatment investigation to confirm diagnosis."
    REQUIRES_MONITORING_TEST = "Safety/Efficacy checks required during active treatment."
    TREATMENT_OPTION = "Intervention (drug, surgery, lifestyle) used for treatment."
    FOLLOW_UP_PLAN = "Scheduled re-evaluation of patient status."
    INDICATED_FOR = "Specific condition a treatment is approved for."
    CONTRAINDICATED_WITH = "Conditions or drugs that prohibit treatment use."
    MANIFESTS_AS = "Clinical signs, symptoms, or phenotypic abnormalities."
    DEMOGRAPHICS = "Patient populations, demographics, or age group."

    @classmethod
    def _missing_(cls, value):
        """
        Redirects Pydantic to look for the Enum name if the value is not found.
        """
        if isinstance(value, str):
            for member in cls:
                # Check if 'INDICATED_FOR' == 'INDICATED_FOR'
                if member.name == value.upper().strip():
                    return member
        return None


class Triplet(BaseModel):
    """
    Triplet of [Entity X, Predicate, Entity Y].
    Example: ["Lisinopril", "FOLLOW_UP", "Kidney function test in 2 weeks"].
    """
    subject_entity: str = Field(description="Source entity.")
    predicate: MedicalPredicate = Field(description="Relationship from source entity to target entity.")
    object_entity: str = Field(description="Target entity.")


class TripletExtraction(BaseModel):
    """
    Collection of all triplets.
    """
    triplets: list[Triplet]
