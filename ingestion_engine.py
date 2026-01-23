from enum import Enum
from pydantic import BaseModel, Field
import json


class MedicalPredicate(Enum):
    """
    These are the seven predicates that are necessary and sufficient to handle pairwise entity relationships.
    """
    REQUIRES_DIAGNOSTIC_TEST = "Pre-treatment investigation to confirm diagnosis."
    REQUIRES_MONITORING_TEST = "Safety/Efficacy checks required during active treatment."
    TREATMENT_OPTION = "Intervention (drug, surgery, lifestyle) used for treatment."
    FOLLOW_UP_PLAN = "Scheduled re-evaluation of patient status."
    INDICATED_FOR = "Specific condition a treatment is approved for."
    CONTRAINDICATED_WITH = "Conditions or drugs that prohibit treatment use."
    MANIFESTS_AS = "Clinical signs, symptoms, or phenotypic abnormalities."


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


class IngestionEngine:
    """
    Engine that handles the transition from text to graph triplets.
    """
    def __init__(self, client, model) -> None:
        self._client = client
        self._model = model


    def _llm_call(self, system_prompt: str, user_content: str):
        """
        Helper to handle API calls with JSON formatting.
        """
        try:
            response = self._client.chat.complete(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return None


    def extract_entities(self, batch_text: list[str]) -> list[list[dict]]:
        """
        Perform batch Named Entity Recognition (NER) on text segments to identify medical concepts.
        Preliminary pass to extract entities, aiming to achieve a high recall.

        Args:
            batch_text: List of independent text segments/chunks to be processed.

        Returns:
            Nested list where each outer element corresponds to an input segment by index. Each inner list contains
            dictionaries representing the extracted entities, e.g., [{"name": "Pylera", "type": "Drug"}].
        """

        system_prompt = """
        You are a specialized medical information extractor.
        You will receive a numbered list of text segments.
        Extract medical entities (Diseases, Drugs, Tests) for each segment separately.

        OUTPUT FORMAT:
        Return a JSON object with a key "results" containing a list of objects.
        Each object must have:
        - "segment_id": The number of the text segment.
        - "entities": A list of {"name": "...", "type": "..."}.
        """
        user_content = "\n\n".join([f"SEGMENT {i}:\n{text}" for i, text in enumerate(batch_text)])

        response = self._llm_call(system_prompt=system_prompt, user_content=user_content)

        # Prevent crashes if "results" is missing
        if not response or "results" not in response:
            print("WARNING: Entity extraction returned no results.")
            return [[] for _ in range(len(batch_text))]

        results_map = {item['segment_id']: item['entities'] for item in response.get('results', [])}
        results = [results_map.get(i, []) for i in range(len(batch_text))]

        return results


    def resolve_entities(self, entities: list[dict]) -> list[dict]:
        """
        Perform Entity Resolution (ER) to map medical variations to canonical forms.
        Avoid synonyms and acronyms, semantically regroup entities (e.g., "Hypertension", "HTN", "High Blood Pressure").

        Args:
            entities: List of extracted entity dictionaries to normalize.

        Returns:
            List of normalized entity dictionaries of the same length and order as the input.
        """

        system_prompt = """
        You are a medical data architect specializing in Entity Resolution.
        For EVERY entity provided in the input list, you must provide its canonical version.

        RULES:
        1. Do not consolidate the list. If I send 10 entities, you must return exactly 10 entities in the same order.
        2. For synonyms (e.g., 'HTA'), provide the standard French term (e.g., 'Hypertension').
        3. Keep the clinical category (Disease, Drug, etc.) accurate.

        OUTPUT FORMAT:
        Return a JSON object with a key 'normalized_entities' containing the list.
        Example:
        Input: [{"name": "HTA"}, {"name": "Stroke"}]
        Output: {"normalized_entities": [{"name": "Hypertension", "type": "Disease"}, {"name": "Accident Vasculaire Cérébral", "type": "Disease"}]}
        """
        user_content = f"Entities to normalize: {json.dumps(entities)}"

        response = self._llm_call(system_prompt, user_content)
        results = response.get('normalized_entities', [])

        # Verification whether the LLM returned the same number of entities that we gave as input.
        # This step is crucial to prevent the routine "zip(stric=True)" from the registry from crashing.
        if len(results) != len(entities):
            print(f"DEBUG: LLM returned {len(results)} but we expected {len(entities)}.")

            # If the LLM returned too few entities, we pad the results with "Unknown".
            while len(results) < len(entities):
                results.append({"name": "Unknown", "type": "Unknown"})

            # If the LLM returned too many (rare), we truncate.
            results = results[:len(entities)]

        return results


    def extract_triplets(self, source_text: str, entities: list[dict]) -> list[dict]:
        """
        Extract directed semantic relationships between validated medical entities.
        For example: ["Lisinopril", "FOLLOW_UP", "Kidney function test in 2 weeks"].

        Args:
            source_text: Raw text segment containing potential relationships.
            entities: List of normalized entities (canonical names) available for relationship mapping.

        Returns:
            List of validated triplets in the format: {"subject": str, "predicate": str, "object": str}.
        """

        # Prepare schema and constraints for the prompt
        predicate_info = "\n".join([f"- {p.name}: {p.value}" for p in MedicalPredicate])

        # Get the JSON schema to show Mistral exactly what we expect
        json_schema = json.dumps(TripletExtraction.model_json_schema(), indent=2)

        system_prompt = f"""
        You are a medical knowledge graph builder. Extract relationships between entities as triplets.

        STRICT DIRECTIONALITY RULES:
        1. INDICATED_FOR: [Drug/Treatment] -> [Disease]
        2. MANIFESTS_AS: [Disease] -> [Symptom/Sign]
        3. CONTRAINDICATED_WITH: [Drug] -> [Condition/Other Drug]
        4. REQUIRES_DIAGNOSTIC_TEST: [Disease] -> [Test]
        5. REQUIRES_MONITORING_TEST: [Drug] -> [Test]
        6. TREATMENT_OPTION: [Disease] -> [Drug/Protocol/Procedure]
        7. FOLLOW_UP_PLAN: [Treatment/Disease] -> [Schedule/Action]

        ALLOWED PREDICATES:
        {predicate_info}

        OUTPUT FORMAT:
        You must return a JSON object that adheres to this schema:
        {json_schema}
        """

        entity_names = [e['name'] for e in entities if 'name' in e]

        user_content = f"""
        TEXT: "{source_text}"

        VALID ENTITIES: {', '.join(entity_names)}

        Instructions:
        1. Only use the provided VALID ENTITIES for 'subject_entity' and 'object_entity'.
        2. Only use the ALLOWED PREDICATES for 'predicate'.
        3. If no relationships are found, return {{"triplets": []}}.
        """

        # Call LLM
        raw_json = self._llm_call(system_prompt, user_content)

        if not raw_json:
            return []

        # Validate and parse the LLM output with Pydantic
        try:
            validated_data = TripletExtraction.model_validate(raw_json)

            # Filter out hallucinations not in the entity_names list
            final_triplets = []
            for triplet in validated_data.triplets:
                if triplet.subject_entity in entity_names and triplet.object_entity in entity_names:
                    final_triplets.append({
                        "subject": triplet.subject_entity,
                        "predicate": triplet.predicate.name,
                        "object": triplet.object_entity
                    })
                else:
                    print(f"DEBUG: Dropping hallucinated entity in triplet: {triplet}")
            return final_triplets

        except Exception as e:
            print(f"Validation Error in Triplets: {e}")
            return []

