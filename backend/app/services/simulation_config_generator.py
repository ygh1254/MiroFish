"""
generation
LLMBased onSimulation requirementcontentgenerationParameters
NoneParameters

groupedgenerationone-shotgenerationcontent
1. generation
2. generation
3. groupedgenerationAgent
4. generation
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger("mirofish.simulation_config")

LOCALE_TIME_CONFIGS = {
    "KR": {
        "label": "한국 (KST)",
        "peak_hours": [20, 21, 22, 23],
        "off_peak_hours": [1, 2, 3, 4, 5, 6],
        "morning_hours": [7, 8, 9],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    },
    "JP": {
        "label": "일본 (JST)",
        "peak_hours": [19, 20, 21, 22],
        "off_peak_hours": [1, 2, 3, 4, 5],
        "morning_hours": [6, 7, 8],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    },
    "US": {
        "label": "미국 (현지 시간 중심)",
        "peak_hours": [19, 20, 21, 22],
        "off_peak_hours": [1, 2, 3, 4, 5],
        "morning_hours": [6, 7, 8],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    },
    "GLOBAL": {
        "label": "글로벌/중립",
        "peak_hours": [18, 19, 20, 21, 22],
        "off_peak_hours": [2, 3, 4, 5, 6],
        "morning_hours": [7, 8, 9],
        "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    },
}


def infer_locale(text: str) -> str:
    lowered = text.lower()
    if any(k in text for k in ["한국", "대한민국", "코스피", "코스닥", "원화"]) or any(
        k in lowered for k in ["kospi", "kosdaq", "krw", "seoul"]
    ):
        return "KR"
    if any(k in text for k in ["일본", "엔화", "닛케이"]) or any(
        k in lowered for k in ["japan", "tokyo", "nikkei", "jpy", "mercari", "rakuten"]
    ):
        return "JP"
    if any(k in text for k in ["미국", "연준", "나스닥", "다우"]) or any(
        k in lowered
        for k in [
            "united states",
            "u.s.",
            "usa",
            "federal reserve",
            "nasdaq",
            "dow",
            "s&p",
        ]
    ):
        return "US"
    return "GLOBAL"


@dataclass
class AgentActivityConfig:
    """singleAgent"""

    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Activity level (0.0-1.0)
    activity_level: float = 0.5  # Activity level

    # hours
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # 24hours0-23
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # minutes
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment tendency (-1.01.0)
    sentiment_bias: float = 0.0

    # 
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # InfluenceOtherAgent
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """"""

    # hours
    total_simulation_hours: int = 72  # default72hours3

    # roundsminutes- default60minutes1hours
    minutes_per_round: int = 60

    # hoursAgent
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak period19-22
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Off-peak period0-5None
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Activity level

    # Morning period
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Work period
    work_hours: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """"""

    # Start
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # 
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # 
    hot_topics: List[str] = field(default_factory=list)

    # 
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """"""

    platform: str  # twitter or reddit

    # 
    recency_weight: float = 0.4  # 
    popularity_weight: float = 0.3  # 
    relevance_weight: float = 0.3  # 

    # after
    viral_threshold: int = 10

    # 
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """completeParameters"""

    # 
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # 
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Agentlist
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # 
    event_config: EventConfig = field(default_factory=EventConfig)

    # 
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLM
    llm_model: str = ""
    llm_base_url: str = ""

    # generationdata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # LLM

    def to_dict(self) -> Dict[str, Any]:
        """"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config)
            if self.twitter_config
            else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """JSONcharacters"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    generation

    LLMgroupedSimulation requirementcontent
    generationParameters

    groupedgeneration
    1. generation
    2. groupedgenerationAgent10-20
    3. generation
    """

    # characters
    MAX_CONTEXT_LENGTH = 50000
    # generationAgent
    AGENTS_PER_BATCH = 15

    # characters
    TIME_CONFIG_CONTEXT_LENGTH = 10000  # 
    EVENT_CONFIG_CONTEXT_LENGTH = 8000  # 
    ENTITY_SUMMARY_LENGTH = 300  # 
    AGENT_SUMMARY_LENGTH = 300  # Agent
    ENTITIES_PER_TYPE_DISPLAY = 20  # 

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY ")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        generationcompletegroupedgeneration

        Args:
            simulation_id: simulation ID
            project_id: Project ID
            graph_id: ID
            simulation_requirement: Simulation requirement
            document_text: content
            entities: afterlist
            enable_twitter: whetherTwitter
            enable_reddit: whetherReddit
            progress_callback: Progress callback(current_step, total_steps, message)

        returns:
            SimulationParameters: completeParameters
        """
        logger.info(
            f"Startgeneration: simulation_id={simulation_id}, ={len(entities)}"
        )

        # 
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  #  +  + NAgent + 
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. build
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
        )

        reasoning_parts = []

        # ========== Step 1: time configuration ==========
        report_progress(1, "시간 설정 생성 중...")
        num_entities = len(entities)
        locale_code = infer_locale(f"{simulation_requirement}\n{context}")
        time_config_result = self._generate_time_config(
            context, num_entities, locale_code
        )
        time_config = self._parse_time_config(
            time_config_result, num_entities, locale_code
        )
        reasoning_parts.append(
            f"시간 설정: {time_config_result.get('reasoning', '성공')}"
        )

        # ========== Step 2: event configuration ==========
        report_progress(2, "이벤트 설정과 핵심 토픽 생성 중...")
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"이벤트 설정: {event_config_result.get('reasoning', '성공')}"
        )

        # ========== Step 3-N: generate agent configs in batches ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Agent 설정 생성 중 ({start_idx + 1}-{end_idx}/{len(entities)})...",
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
                locale_code=locale_code,
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(f"Agent 설정: {len(all_agent_configs)}개 생성 완료")

        # ========== Assign posters for initial posts ==========
        logger.info("초기 게시물 발행 Agent를 배정하는 중...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len(
            [
                p
                for p in event_config.initial_posts
                if p.get("poster_agent_id") is not None
            ]
        )
        reasoning_parts.append(
            f"초기 게시물 배정: {assigned_count}개 게시물에 발행자를 연결함"
        )

        # ========== after: generation ==========
        report_progress(total_steps, "generation...")
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5,
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6,
            )

        # buildParameters
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts),
        )

        logger.info(f"generationcomplete: {len(params.agent_configs)} Agent")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
    ) -> str:
        """buildLLM"""

        # 
        entity_summary = self._summarize_entities(entities)

        # build
        context_parts = [
            f"## Simulation requirement\n{simulation_requirement}",
            f"\n##  ({len(entities)})\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = (
            self.MAX_CONTEXT_LENGTH - current_length - 500
        )  # 500characters

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...()"
            context_parts.append(f"\n## content\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """generation"""
        lines = []

        # typesgrouped
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # 
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (
                    (e.summary[:summary_len] + "...")
                    if len(e.summary) > summary_len
                    else e.summary
                )
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... has {len(type_entities) - display_count} ")

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLMCallJSON"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    # response_format removed for Codex compat
                    temperature=0.7 - (attempt * 0.1),  # 
                    # max_tokensLLM
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Checkwhether
                if finish_reason == "length":
                    logger.warning(f"LLMoutput (attempt {attempt + 1})")
                    content = self._fix_truncated_json(content)

                # JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON (attempt {attempt + 1}): {str(e)[:80]}"
                    )

                    # JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLMCall (attempt {attempt + 1}): {str(e)[:80]}")
                last_error = e
                import time

                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLMCall")

    def _fix_truncated_json(self, content: str) -> str:
        """JSON"""
        content = content.strip()

        # 
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Checkwhetherhascharacters
        if content and content[-1] not in '",}]':
            content += '"'

        # 
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """JSON"""
        import re

        # 
        content = self._fix_truncated_json(content)

        # JSONgrouped
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # characters
            def fix_string(match):
                s = match.group(0)
                s = s.replace("\n", " ").replace("\r", " ")
                s = re.sub(r"\s+", " ", s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # hascharacters
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                json_str = re.sub(r"\s+", " ", json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(
        self, context: str, num_entities: int, locale_code: str
    ) -> Dict[str, Any]:
        """generation"""
        # 
        context_truncated = context[: self.TIME_CONFIG_CONTEXT_LENGTH]

        # 80%agent
        max_agents_allowed = max(1, int(num_entities * 0.9))

        locale_cfg = LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS["GLOBAL"])

        prompt = f"""Simulation requirementgeneration

{context_truncated}

## 
generationJSON

### Based on
-  locale: {locale_cfg["label"]}
- Active period locale 
- /
- ****YouBased on
  - 21-23
  - hasoff_peak_hours 

### returnsJSONmarkdown


{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": {locale_cfg["peak_hours"]},
    "off_peak_hours": {locale_cfg["off_peak_hours"]},
    "morning_hours": {locale_cfg["morning_hours"]},
    "work_hours": {locale_cfg["work_hours"]},
    "reasoning": " locale "
}}


- total_simulation_hours (int): 24-168hours
- minutes_per_round (int): Minutes per round30-120minutes60minutes
- agents_per_hour_min (int): hoursAgent: 1-{max_agents_allowed}
- agents_per_hour_max (int): hoursAgent: 1-{max_agents_allowed}
- peak_hours (int): Peak periodBased on
- off_peak_hours (int): Off-peak period
- morning_hours (int): Morning period
- work_hours (int): Work period
- reasoning (string): """

        system_prompt = f"YoureturnsJSON locale{locale_cfg['label']}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLMgeneration: {e}, default")
            return self._get_default_time_config(num_entities, locale_code)

    def _get_default_time_config(
        self, num_entities: int, locale_code: str
    ) -> Dict[str, Any]:
        """Getdefaultlocale-aware"""
        locale_cfg = LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS["GLOBAL"])
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # rounds1hours
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": locale_cfg["peak_hours"],
            "off_peak_hours": locale_cfg["off_peak_hours"],
            "morning_hours": locale_cfg["morning_hours"],
            "work_hours": locale_cfg["work_hours"],
            "reasoning": f"기본 locale 설정({locale_cfg['label']})을 사용한 시간 구성",
        }

    def _parse_time_config(
        self, result: Dict[str, Any], num_entities: int, locale_code: str
    ) -> TimeSimulationConfig:
        """agents_per_houragent"""
        locale_cfg = LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS["GLOBAL"])
        # Get
        agents_per_hour_min = result.get(
            "agents_per_hour_min", max(1, num_entities // 15)
        )
        agents_per_hour_max = result.get(
            "agents_per_hour_max", max(5, num_entities // 5)
        )

        # agent
        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) Agent ({num_entities})"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) Agent ({num_entities})"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        #  min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(
                f"agents_per_hour_min >= max {agents_per_hour_min}"
            )

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # defaultrounds1hours
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", locale_cfg["peak_hours"]),
            off_peak_hours=result.get("off_peak_hours", locale_cfg["off_peak_hours"]),
            off_peak_activity_multiplier=0.05,  # None
            morning_hours=result.get("morning_hours", locale_cfg["morning_hours"]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", locale_cfg["work_hours"]),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5,
        )

    def _generate_event_config(
        self, context: str, simulation_requirement: str, entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """generation"""

        # Gettypeslist LLM 
        entity_types_available = list(
            set(e.get_entity_type() or "Unknown" for e in entities)
        )

        # types
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join(
            [f"- {t}: {', '.join(examples)}" for t, examples in type_examples.items()]
        )

        # 
        context_truncated = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Simulation requirementgeneration

Simulation requirement: {simulation_requirement}

{context_truncated}

## types
{type_info}

## 
generationJSON
- 
- 
- content**each poster_typetypes**

****: poster_type "types"only thengrouped Agent 
 Official/University types MediaOutlet  Student 

returnsJSONmarkdown
{{
    "hot_topics": ["1", "2", ...],
    "narrative_direction": "<>",
    "initial_posts": [
        {{"content": "content", "poster_type": "typestypes"}},
        ...
    ],
    "reasoning": "<>"
}}"""

        system_prompt = "YougroupedreturnsJSON poster_type types"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLMgeneration: {e}, default")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "default",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", ""),
        )

    def _assign_initial_post_agents(
        self, event_config: EventConfig, agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        grouped Agent

        Based oneach poster_type  agent_id
        """
        if not event_config.initial_posts:
            return event_config

        # types agent 
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # types LLM outputdifferent
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # types agent  agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            #  agent
            matched_agent_id = None

            # 1. 
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. 
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. Not foundInfluence agent
            if matched_agent_id is None:
                logger.warning(
                    f"Not foundtypes '{poster_type}'  AgentInfluence Agent"
                )
                if agent_configs:
                    # InfluenceInfluence
                    sorted_agents = sorted(
                        agent_configs, key=lambda a: a.influence_weight, reverse=True
                    )
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append(
                {
                    "content": content,
                    "poster_type": post.get("poster_type", "Unknown"),
                    "poster_agent_id": matched_agent_id,
                }
            )

            logger.info(
                f"grouped: poster_type='{poster_type}' -> agent_id={matched_agent_id}"
            )

        event_config.initial_posts = updated_posts
        return event_config

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str,
        locale_code: str,
    ) -> List[AgentActivityConfig]:
        """groupedgenerationAgent"""

        # build
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append(
                {
                    "agent_id": start_idx + i,
                    "entity_name": e.name,
                    "entity_type": e.get_entity_type() or "Unknown",
                    "summary": e.summary[:summary_len] if e.summary else "",
                }
            )

        prompt = f"""eachgeneration

Simulation requirement: {simulation_requirement}

## list
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

        ## 
        eachgeneration
        - ** locale{LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS["GLOBAL"])["label"]}**
        - ****University/GovernmentAgencyActivity level(0.1-0.3)(9-17)(60-240minutes)Influence(2.5-3.0)
        - ****MediaOutletActivity level(0.4-0.6)(8-23)(5-30minutes)Influence(2.0-2.5)
        - ****Student/Person/AlumniActivity level(0.6-0.9)(18-23)(1-15minutes)Influence(0.8-1.2)
- **/**Activity level(0.4-0.6)Influence(1.5-2.0)

returnsJSONmarkdown
{{
    "agent_configs": [
        {{
            "agent_id": <>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <>,
            "comments_per_hour": <>,
            "active_hours": [<hourslist locale >],
            "response_delay_min": <Response delayminutes>,
            "response_delay_max": <Response delayminutes>,
            "sentiment_bias": <-1.01.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <Influence>
        }},
        ...
    ]
}}"""

        system_prompt = f"YougroupedreturnsJSON locale{LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS['GLOBAL'])['label']}"

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
            }
        except Exception as e:
            logger.warning(f"AgentLLMgeneration: {e}, generation")
            llm_configs = {}

        # buildAgentActivityConfig
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # LLMhasgenerationgeneration
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity, locale_code)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0),
            )
            configs.append(config)

        return configs

    def _generate_agent_config_by_rule(
        self, entity: EntityNode, locale_code: str
    ) -> Dict[str, Any]:
        """generationsingleAgentlocale-aware"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        locale_cfg = LOCALE_TIME_CONFIGS.get(locale_code, LOCALE_TIME_CONFIGS["GLOBAL"])

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": locale_cfg["work_hours"],
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0,
            }
        elif entity_type in ["mediaoutlet"]:
            # Influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": sorted(
                    set(
                        locale_cfg["morning_hours"]
                        + locale_cfg["work_hours"]
                        + locale_cfg["peak_hours"]
                        + [23]
                    )
                ),
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5,
            }
        elif entity_type in ["professor", "expert", "official"]:
            # /+
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": sorted(
                    set(locale_cfg["work_hours"] + locale_cfg["peak_hours"])
                ),
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0,
            }
        elif entity_type in ["student"]:
            # 
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": sorted(
                    set(locale_cfg["morning_hours"] + locale_cfg["peak_hours"])
                ),
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8,
            }
        elif entity_type in ["alumni"]:
            # 
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": sorted(set([12, 13] + locale_cfg["peak_hours"])),
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
        else:
            # 
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": sorted(
                    set([9, 10, 11, 12, 13] + locale_cfg["peak_hours"])
                ),
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
