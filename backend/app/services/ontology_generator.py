"""
Ontology generation
API 1: build ontology definitions from source documents
"""

from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


ONTOLOGY_SYSTEM_PROMPT = """You are an ontology design expert.

Read the provided source material and simulation requirement, then generate a compact ontology definition for graph construction.

Requirements:
- Return JSON only. Do not output extra prose.
- Produce exactly 10 entity types.
- Produce 6 to 10 edge types.
- Each entity type must include:
  - name (PascalCase)
  - description
  - attributes (1 to 3 custom attributes)
  - examples
- Do not redefine built-in fields such as name, uuid, group_id, created_at, or summary.
- Use source_targets on each edge type to declare valid source/target pairs.
- Prefer concrete, practical types derived from the material.
- Include a short Korean analysis_summary.

Return JSON in this shape:
{
  "entity_types": [
    {
      "name": "EntityType",
      "description": "Description",
      "attributes": [
        {
          "name": "attribute_name",
          "type": "text",
          "description": "Description"
        }
      ],
      "examples": ["Example 1", "Example 2"]
    }
  ],
  "edge_types": [
    {
      "name": "EDGE_TYPE",
      "description": "Description",
      "source_targets": [
        {"source": "EntityType", "target": "OtherEntityType"}
      ],
      "attributes": []
    }
  ],
  "analysis_summary": "한국어 요약"
}
"""


class OntologyGenerator:
    """
    Ontology generation
    groupedcontentgenerationtypesdefinition
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ontology definition

        Args:
            document_texts: list
            simulation_requirement: Simulation requirement
            additional_context: 

        returns:
            definitionentity_types, edge_types
        """
        # builduser message
        user_message = self._build_user_message(
            document_texts, simulation_requirement, additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # CallLLM
        result = self.llm_client.chat_json(
            messages=messages, temperature=0.3, max_tokens=4096
        )

        # after
        result = self._validate_and_process(result)

        return result

    #  LLM 5
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
    ) -> str:
        """builduser message"""

        # 
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # 5LLMcontentGraph build
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[: self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...({original_length}{self.MAX_TEXT_LENGTH_FOR_LLM}used togrouped)..."

        message = f"""## Simulation requirement

{simulation_requirement}

## content

{combined_text}
"""

        if additional_context:
            message += f"""
## 

{additional_context}
"""

        message += """
Based oncontenttypestypes

****
1. output10 entity types
2. after2typesPerson Organization
3. 8Based oncontenttypes
4. hastypes
5.  nameuuidgroup_id  full_nameorg_name 
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """after"""

        # 
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # description100characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep API  10 Customtypes 10 Customtypes
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # typesdefinition
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {
                    "name": "full_name",
                    "type": "text",
                    "description": "Full name of the person",
                },
                {"name": "role", "type": "text", "description": "Role or occupation"},
            ],
            "examples": ["ordinary citizen", "anonymous netizen"],
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {
                    "name": "org_name",
                    "type": "text",
                    "description": "Name of the organization",
                },
                {
                    "name": "org_type",
                    "type": "text",
                    "description": "Type of organization",
                },
            ],
            "examples": ["small business", "community group"],
        }

        # Checkwhetherhastypes
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # addtypes
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # addafter 10 hastypes
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # types
                result["entity_types"] = result["entity_types"][:-to_remove]

            # addtypes
            result["entity_types"].extend(fallbacks_to_add)

        # 
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        definitionPythonontology.py

        Args:
            ontology: definition

        returns:
            Pythoncharacters
        """
        code_lines = [
            '"""',
            "Customtypesdefinition",
            "MiroFishgenerationused to",
            '"""',
            "",
            "from pydantic import Field",
            "from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel",
            "",
            "",
            "# ============== typesdefinition ==============",
            "",
        ]

        # generationtypes
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f"class {name}(EntityModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        code_lines.append("# ============== typesdefinition ==============")
        code_lines.append("")

        # generationtypes
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # PascalCase
            class_name = "".join(word.capitalize() for word in name.split("_"))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f"class {class_name}(EdgeModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        # generationtypes
        code_lines.append("# ============== types ==============")
        code_lines.append("")
        code_lines.append("ENTITY_TYPES = {")
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append("}")
        code_lines.append("")
        code_lines.append("EDGE_TYPES = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = "".join(word.capitalize() for word in name.split("_"))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append("}")
        code_lines.append("")

        # generationsource_targets
        code_lines.append("EDGE_SOURCE_TARGETS = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ", ".join(
                    [
                        f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                        for st in source_targets
                    ]
                )
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append("}")

        return "\n".join(code_lines)
