"""
Report Agent
LangChain + ZepReACTreportgeneration

feature
1. Based onSimulation requirementZepgenerationreport
2. planningaftergroupedgeneration
3. ReACTrounds
4. Calltool
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult,
)

logger = get_logger("mirofish.report_agent")


class ReportLogger:
    """
    Report Agent logs

    reportgeneration agent_log.jsonl 
    complete JSON typescontent
    """

    def __init__(self, report_id: str):
        """
        Initializelogs

        Args:
            report_id: reportIDused tologspath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "agent_log.jsonl"
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """logs"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """GetStart"""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None,
    ):
        """
        logs

        Args:
            action: types 'start', 'tool_call', 'llm_response', 'section_complete' 
            stage:  'planning', 'generating', 'completed'
            details: content
            section_title: sectionoptional
            section_index: sectionoptional
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details,
        }

        #  JSONL 
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """reportgenerationStart"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "보고서 생성 작업 시작",
            },
        )

    def log_planning_start(self):
        """planningStart"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "보고서 개요 구성 시작"},
        )

    def log_planning_context(self, context: Dict[str, Any]):
        """planningGet"""
        self.log(
            action="planning_context",
            stage="planning",
            details={"message": "시뮬레이션 컨텍스트 조회", "context": context},
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """planningcomplete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={"message": "개요 구성 완료", "outline": outline_dict},
        )

    def log_section_start(self, section_title: str, section_index: int):
        """sectiongenerationStart"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"섹션 생성 시작: {section_title}"},
        )

    def log_react_thought(
        self, section_title: str, section_index: int, iteration: int, thought: str
    ):
        """ ReACT """
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT {iteration}차 사고",
            },
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int,
    ):
        """toolCall"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"도구 호출: {tool_name}",
            },
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int,
    ):
        """toolCallcontent"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # 
                "result_length": len(result),
                "message": f"도구 {tool_name} 결과 반환",
            },
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool,
    ):
        """ LLM content"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # 
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM 응답 (도구 호출: {has_tool_calls}, 최종 답변: {has_final_answer})",
            },
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int,
    ):
        """sectioncontentgenerationcompletecontentsectioncomplete"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # content
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"섹션 {section_title} 본문 생성 완료",
            },
        )

    def log_section_full_complete(
        self, section_title: str, section_index: int, full_content: str
    ):
        """
        sectiongenerationcomplete

        frontendlogsdeterminesectionwhethercompleteGetcontent
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"섹션 {section_title} 생성 완료",
            },
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """reportgenerationcomplete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "보고서 생성 완료",
            },
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={"error": error_message, "message": f"오류 발생: {error_message}"},
        )


class ReportConsoleLogger:
    """
    Report Agent consolelogs

    consolelogsINFOWARNINGreport console_log.txt 
    logs agent_log.jsonl differentplain-text formatconsoleoutput
    """

    def __init__(self, report_id: str):
        """
        Initializeconsolelogs

        Args:
            report_id: reportIDused tologspath
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "console_log.txt"
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        """logs"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """logs"""
        import logging

        # create
        self._file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.INFO)

        # console
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self._file_handler.setFormatter(formatter)

        # add report_agent  logger
        loggers_to_attach = [
            "mirofish.report_agent",
            "mirofish.zep_tools",
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # add
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """ logger """
        import logging

        if self._file_handler:
            loggers_to_detach = [
                "mirofish.report_agent",
                "mirofish.zep_tools",
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """"""
        self.close()


class ReportStatus(str, Enum):
    """reportstate"""

    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """reportsection"""

    title: str
    content: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "description": self.description,
        }

    def to_markdown(self, level: int = 2) -> str:
        """Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """report"""

    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        """Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """report"""

    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════
# Prompt 
# ═══════════════════════════════════════════════════════════════

# ── tool ──

TOOL_DESC_INSIGHT_FORGE = """\
[심층 인사이트 검색 - 핵심 분석 도구]
복합적인 주제를 깊게 파고들 때 사용하는 핵심 검색 도구다. 이 도구는:
1. 질문을 여러 하위 질문으로 분해하고
2. 시뮬레이션 그래프를 다양한 각도에서 검색하며
3. 의미 검색, 엔티티 분석, 관계 추적 결과를 통합하고
4. 가장 풍부하고 깊이 있는 근거를 반환한다

[이럴 때 사용]
- 특정 주제를 깊게 분석해야 할 때
- 사건의 여러 측면을 함께 이해해야 할 때
- 보고서 섹션을 뒷받침할 근거가 충분히 필요할 때

[반환 내용]
- 바로 인용 가능한 관련 사실 원문
- 핵심 엔티티 인사이트
- 관계망 분석"""

TOOL_DESC_PANORAMA_SEARCH = """\
[전방위 검색 - 전체 흐름 파악]
시뮬레이션 결과의 전체 그림과 전개 흐름을 볼 때 적합한 도구다. 이 도구는:
1. 관련 노드와 관계를 폭넓게 수집하고
2. 현재 유효한 사실과 과거/만료된 사실을 구분하며
3. 여론과 사건 흐름이 어떻게 전개됐는지 보여준다

[이럴 때 사용]
- 사건의 전체 전개 맥락이 필요할 때
- 단계별 변화 양상을 비교하고 싶을 때
- 엔티티와 관계 정보를 넓게 파악해야 할 때

[반환 내용]
- 현재 유효한 사실(최신 시뮬레이션 결과)
- 과거/만료 사실(변화 이력)
- 관련된 전체 엔티티"""

TOOL_DESC_QUICK_SEARCH = """\
[빠른 검색 - 즉시 확인용]
간단하고 직접적인 정보를 빠르게 확인할 때 쓰는 경량 검색 도구다.

[이럴 때 사용]
- 특정 정보를 빠르게 찾고 싶을 때
- 어떤 사실을 즉시 검증하고 싶을 때
- 단순한 정보 조회가 필요할 때

[반환 내용]
- 질의와 가장 관련성이 높은 사실 목록"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[심층 인터뷰 - 실제 Agent 인터뷰(양 플랫폼)]
실행 중인 OASIS 시뮬레이션 Agent를 실제 인터뷰 API로 호출해 답변을 수집한다.
이건 LLM이 상상해서 만든 답이 아니라, 시뮬레이션 Agent의 실제 응답이다.
기본적으로 Twitter와 Reddit 양쪽에서 동시에 인터뷰해 더 입체적인 관점을 확보한다.

동작 방식:
1. 페르소나 파일을 읽어 전체 Agent 구성을 파악하고
2. 인터뷰 주제와 가장 관련 있는 Agent를 고르며
3. 인터뷰 질문을 자동 생성하고
4. /api/simulation/interview/batch 로 양 플랫폼 인터뷰를 수행한 뒤
5. 결과를 통합해 다각도 분석을 제공한다

[이럴 때 사용]
- 서로 다른 역할의 시각을 비교하고 싶을 때
- 다양한 입장과 의견을 수집해야 할 때
- 시뮬레이션 Agent의 실제 응답이 필요할 때
- 보고서에 인터뷰 인용과 생생한 관찰을 넣고 싶을 때

[반환 내용]
- 인터뷰 대상 Agent의 정체성 정보
- 각 Agent의 Twitter/Reddit 인터뷰 답변
- 바로 인용 가능한 핵심 발언
- 인터뷰 요약과 관점 비교

[중요] OASIS 시뮬레이션 환경이 실행 중이어야 사용할 수 있다!"""

# ── planning prompt ──

PLAN_SYSTEM_PROMPT = """\
너는 미래 예측 보고서를 작성하는 전문가다. 너는 시뮬레이션 세계를 위에서 내려다보는 관찰자처럼 각 Agent의 행동, 발언, 상호작용을 읽어낼 수 있다.

[핵심 관점]
우리는 시뮬레이션 세계에 특정한 조건과 요구사항을 주입했다. 그 결과로 나타난 변화는 미래에 일어날 수 있는 상황에 대한 예측이다. 네가 보는 것은 단순한 실험 데이터가 아니라 미래의 예행연습이다.

[너의 임무]
미래 예측 보고서를 작성해 다음 질문에 답하라.
1. 주어진 조건에서 미래에는 어떤 일이 벌어졌는가?
2. 각 Agent 집단은 어떻게 반응하고 행동했는가?
3. 이 시뮬레이션은 어떤 미래 흐름과 위험 신호를 드러내는가?

[보고서 방향]
- 이 보고서는 시뮬레이션 기반의 미래 예측 보고서다
- 사건 전개, 집단 반응, 창발 현상, 잠재 리스크 같은 예측 결과에 집중한다
- 시뮬레이션 속 Agent의 행동과 발언은 미래 집단행동의 단서다
- 현실 현황을 해설하는 일반 분석문처럼 쓰지 마라
- 피상적인 동향 요약으로 흐르지 마라

[언어 요구사항]
- 최종 출력은 제목, 요약, 섹션 제목과 설명까지 모두 자연스럽고 전문적인 한국어로 작성한다

[섹션 수 제한]
- 섹션은 최소 2개, 최대 5개다
- 하위 섹션은 만들지 말고 각 섹션을 하나의 완결된 단위로 설계한다
- 내용은 군더더기 없이 핵심 예측 발견에 집중한다
- 각 섹션은 서로 다른 분석 임무를 가져야 하며 내용이 겹치면 안 된다
- 시간 흐름, 반응 주체, 구조적 위험처럼 관점을 분리해 설계하라
- 시뮬레이션 요구사항에 국가/권역 비교가 있으면 섹션 구조에도 반영하라

다음 JSON 형식으로 보고서 개요를 출력하라.
{
    "title": "보고서 제목",
    "summary": "핵심 예측 발견을 한 문장으로 요약",
    "sections": [
        {
            "title": "섹션 제목",
            "description": "섹션에서 다룰 내용"
        }
    ]
}

주의: sections 배열은 최소 2개, 최대 5개의 요소만 포함해야 한다."""

PLAN_USER_PROMPT_TEMPLATE = """\
[예측 시나리오]
시뮬레이션 세계에 주입한 조건(시뮬레이션 요구사항): {simulation_requirement}

[시뮬레이션 규모]
- 참여 엔티티 수: {total_nodes}
- 생성된 관계 수: {total_edges}
- 엔티티 유형 분포: {entity_types}
- 관계 유형 분포: {relation_types}
- 활성 Agent 수: {total_entities}

[대표 엔티티 샘플]
{representative_entities_json}

[예측된 미래 사실 샘플]
{related_facts_json}

이 미래 예행연습을 관찰자 시점에서 검토하라.
1. 주어진 조건 아래 미래는 어떤 상태로 전개됐는가?
2. 각 Agent 집단은 어떻게 반응하고 행동했는가?
3. 이 시뮬레이션은 어떤 주목할 만한 미래 흐름을 보여주는가?
4. 어느 국가/권역/집단의 시각이 서로 다르게 갈리는가?

예측 결과를 바탕으로 가장 적합한 보고서 섹션 구조를 설계하라.

[다시 강조]
- 섹션 수는 최소 2개, 최대 5개
- 각 섹션은 서로 다른 질문에 답해야 하며 핵심 근거도 달라야 함
- 같은 현상을 반복 설명하지 말고 시간, 집단, 시스템 리스크처럼 초점을 분리하라
- 제목, 요약, 섹션 제목과 설명은 모두 한국어로 작성"""

# ── sectiongeneration prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
너는 미래 예측 보고서의 한 섹션을 작성하는 전문가다. 최종 출력은 반드시 자연스럽고 전문적인 한국어여야 한다.

보고서 제목: {report_title}
보고서 요약: {report_summary}
예측 시나리오(시뮬레이션 요구사항): {simulation_requirement}

현재 작성할 섹션: {section_title}
섹션 설계 의도: {section_description}

═══════════════════════════════════════════════════════════════
[핵심 관점]
═══════════════════════════════════════════════════════════════

시뮬레이션 세계는 미래의 예행연습이다. 우리는 특정 조건을 주입했고,
그 안에서 나타난 Agent의 행동과 상호작용은 미래 집단행동의 예측 신호다.

너의 임무는 다음과 같다.
- 주어진 조건에서 미래에 어떤 일이 일어났는지 드러낸다
- 각 Agent 집단이 어떻게 반응하고 움직였는지 설명한다
- 주목할 만한 미래 흐름, 리스크, 기회를 찾아낸다

❌ 현실 현황을 설명하는 분석문처럼 쓰지 마라
✅ "앞으로 어떻게 될 것인가"라는 관점에 집중하라

═══════════════════════════════════════════════════════════════
[반드시 지켜야 할 규칙]
═══════════════════════════════════════════════════════════════

1. [반드시 도구로 시뮬레이션 세계를 관찰할 것]
   - 너는 관찰자 시점에서 미래의 예행연습을 읽고 있다
   - 모든 내용은 시뮬레이션 안에서 실제로 일어난 사건과 Agent의 발언/행동에 근거해야 한다
   - 너 자신의 일반 지식으로 내용을 채우면 안 된다
   - 각 섹션마다 최소 1회는 도구를 호출해 근거를 확보하고, 충분한 근거가 모이면 즉시 마무리한다

2. [반드시 Agent의 원문 발언과 행동을 근거로 인용할 것]
   - Agent의 발언과 행동은 미래 집단행동의 예측 단서다
   - 보고서에서는 인용 형식으로 그 근거를 보여줘야 한다
   - 이러한 인용은 보고서의 핵심 증거다

3. [언어 일관성]
   - 도구 결과에는 영어 또는 혼합 언어가 포함될 수 있다
   - 시뮬레이션 요구사항이나 원문 언어와 무관하게 보고서는 모두 한국어로 작성한다
   - 인용이나 본문에 들어갈 외국어 내용은 의미를 유지한 채 자연스러운 한국어로 번역한다
   - 영어 엔티티/집단명은 첫 언급 시 한국어 설명을 먼저 쓰고 필요하면 원문을 괄호로 병기한다

4. [요구사항 커버리지]
   - 시뮬레이션 요구사항에 국가/권역 비교가 있으면 본문에서 빠뜨리지 마라
   - 특정 한 국가 시각만 반복하지 말고, 최소 두 개 이상의 국가/권역 시각 차이를 드러내라
   - 시간축, 집단 반응, 구조적 리스크 중 현재 섹션에 필요한 축을 명확히 선택해 깊게 파고들어라

4. [예측 결과에 충실할 것]
   - 보고서는 시뮬레이션이 보여준 미래 결과를 충실히 반영해야 한다
   - 시뮬레이션에 없는 정보를 추가하지 마라
   - 정보가 부족한 부분은 부족하다고 명시하라

═══════════════════════════════════════════════════════════════
[형식 규칙]
═══════════════════════════════════════════════════════════════

[한 섹션 = 최소 작성 단위]
- 각 섹션은 보고서의 최소 단위다
- ❌ 섹션 내부에 Markdown 제목(#, ##, ###, ####)을 쓰지 마라
- ❌ 본문 시작 부분에 섹션 제목을 다시 쓰지 마라
- ✅ 섹션 제목은 시스템이 자동으로 붙이므로 순수 본문만 작성한다
- ✅ **굵은 글씨**, 단락, 인용, 목록으로 구조를 잡되 제목은 사용하지 않는다

[올바른 예시]
```
이 섹션은 사건의 여론 확산 양상을 분석한다. 시뮬레이션 데이터를 종합하면 다음과 같은 흐름이 보인다.

**초기 확산 단계**

첫 번째 반응은 특정 채널에서 빠르게 증폭되며 출발했다.

> "핵심 참여자들은 초기 신호를 빠르게 확대 재생산했다."

**감정 증폭 단계**

이후 다른 채널에서 감정적 해석이 더 강하게 확산됐다.

- 시각적 자극이 강함
- 감정적 공명도가 높음
```

[잘못된 예시]
```
## 실행 요약
### 1. 초기 단계
#### 1.1 세부 분석

이 섹션은 ...
```

═══════════════════════════════════════════════════════════════
[사용 가능한 검색 도구] (섹션당 1~4회, 필요시 호출)
═══════════════════════════════════════════════════════════════

{tools_description}

[도구 사용 권장]
- insight_forge: 질문을 분해해 다각도로 깊게 검색
- panorama_search: 사건 전체 흐름, 타임라인, 변화 과정 파악
- quick_search: 특정 사실의 빠른 검증
- interview_agents: 다양한 역할의 1인칭 관점과 실제 반응 수집

═══════════════════════════════════════════════════════════════
[작업 방식]
═══════════════════════════════════════════════════════════════

각 응답에서는 아래 둘 중 하나만 할 수 있다.

옵션 A - 도구 호출:
생각을 적은 뒤 아래 형식으로 도구 하나를 호출한다.
<tool_call>
{{"name": "도구명", "parameters": {{"파라미터명": "값"}}}}
</tool_call>
시스템이 도구를 실행해 결과를 돌려준다. 도구 결과를 스스로 지어내면 안 된다.

옵션 B - 최종 본문 출력:
충분한 정보를 모았으면 "Final Answer:"로 시작해 섹션 본문을 출력한다.

⚠️ 절대 금지:
- 한 응답 안에 도구 호출과 Final Answer를 동시에 넣지 마라
- Observation을 지어내지 마라. 모든 도구 결과는 시스템이 주입한다
- 한 번에 도구는 하나만 호출한다

═══════════════════════════════════════════════════════════════
[섹션 내용 요구사항]
═══════════════════════════════════════════════════════════════

1. 내용은 반드시 도구로 수집한 시뮬레이션 데이터에 근거해야 한다
2. 핵심 근거는 충분히 인용해 보여준다
3. Markdown은 사용할 수 있지만 제목 문법은 금지한다
   - **굵은 글씨**로 소주제를 표시할 수 있다
   - 목록(-, 1. 2. 3.)으로 요점을 정리할 수 있다
   - 단락 사이에는 빈 줄을 둔다
4. 인용은 반드시 독립된 단락으로 배치한다
5. 다른 섹션과의 논리적 연결을 유지한다
6. 아래 완료된 섹션을 읽고 중복 설명을 피한다
7. 다시 강조하지만 제목은 쓰지 말고, 필요하면 **굵은 글씨**로 구조를 잡아라"""

SECTION_USER_PROMPT_TEMPLATE = """\
이전에 완료된 섹션 요약(중복을 피하기 위해 꼭 읽어라):
{previous_section_context}

이미 사용한 핵심 인용/근거(가능하면 재사용하지 마라):
{used_quotes}

═══════════════════════════════════════════════════════════════
[현재 작업] 작성할 섹션: {section_title}
═══════════════════════════════════════════════════════════════

[중요 안내]
1. 위 섹션 요약과 이미 사용한 근거를 읽고 같은 설명과 인용을 반복하지 마라
2. 시작 시 최소 1회는 도구를 호출해 시뮬레이션 데이터를 확보하라
3. 이 섹션만의 새로운 근거, 새로운 집단, 새로운 전환 포인트를 찾아라
4. 보고서 내용은 반드시 검색 결과에서 나와야 하며 네 지식을 끼워 넣지 마라
5. 다른 섹션에서 이미 충분히 다룬 프레임은 짧게 연결만 하고, 새로운 분석 축으로 넘어가라
6. 시뮬레이션 요구사항에 포함된 국가/권역(예: 한국, 미국, 일본, 글로벌) 중 현재 섹션과 관련된 비교를 반드시 반영하라
7. 영어 표현을 그대로 두지 말고, 본문 설명은 자연스러운 한국어로 정리하라

[형식 경고]
- ❌ 어떤 제목도 쓰지 마라 (#, ##, ###, #### 금지)
- ❌ "{section_title}"를 본문 첫 줄에 다시 쓰지 마라
- ✅ 섹션 제목은 시스템이 자동으로 붙인다
- ✅ 본문만 쓰고, 필요하면 **굵은 글씨**로 소주제를 표현하라

이제 시작하라:
1. 이 섹션만의 질문을 먼저 분명히 정하고
2. 이전 섹션과 겹치지 않는 근거를 도구로 수집한 뒤
3. 충분한 근거가 모이면 Final Answer로 본문만 출력하라"""

# ── ReACT  ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (검색 결과):

═══ 도구 {tool_name} 결과 ═══
{result}

═══════════════════════════════════════════════════════════════
현재 도구 호출 {tool_calls_count}/{max_tool_calls}회 (사용한 도구: {used_tools_str}){unused_hint}
- 정보가 충분하면 "Final Answer:"로 시작해 섹션 본문을 출력하라 (위 근거를 반드시 반영)
- 정보가 더 필요하면 도구 하나를 더 호출하라
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[주의] 현재 도구를 {tool_calls_count}회만 호출했다. 최소 {min_tool_calls}회는 필요하다."
    " 도구를 더 호출해 시뮬레이션 근거를 확보한 뒤 Final Answer를 출력하라.{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "현재 도구 호출은 {tool_calls_count}회이며 최소 {min_tool_calls}회가 필요하다."
    " 도구를 호출해 시뮬레이션 근거를 더 수집하라.{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "도구 호출 한도({tool_calls_count}/{max_tool_calls})에 도달했다. 더 이상 도구를 호출할 수 없다."
    '지금까지 확보한 정보만으로 즉시 "Final Answer:" 형식의 섹션 본문을 출력하라.'
)

REACT_UNUSED_TOOLS_HINT = (
    "\n💡 아직 사용하지 않은 도구: {unused_list}. 다른 도구도 써서 관점을 넓혀라"
)

REACT_FORCE_FINAL_MSG = "도구 호출 한도에 도달했다. 이제 바로 Final Answer: 형식으로 섹션 본문을 출력하라."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
너는 간결하고 효율적인 시뮬레이션 예측 보조자다.

[배경]
예측 조건: {simulation_requirement}

[이미 생성된 보고서]
{report_content}

[규칙]
1. 가능하면 위 보고서 내용을 우선 근거로 답하라
2. 장황한 사고 과정을 늘어놓지 말고 바로 답하라
3. 보고서만으로 부족할 때에만 도구를 호출해 추가 근거를 찾는다
4. 답변은 간결하고, 명확하며, 구조적으로 정리한다

[사용 가능한 도구] (필요할 때만, 최대 1~2회)
{tools_description}

[도구 호출 형식]
<tool_call>
{{"name": "tool", "parameters": {{"Parameters": "Parameters"}}}}
</tool_call>

[답변 스타일]
- 짧고 직접적으로 답하라
- 핵심 근거는 > 인용 형식으로 제시하라
- 결론을 먼저 말하고 이유를 덧붙여라"""

CHAT_OBSERVATION_SUFFIX = "\n\n질문에 간결하게 답하라."


# ═══════════════════════════════════════════════════════════════
# ReportAgent 
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - reportgenerationAgent

    ReACTReasoning + Acting
    1. planninggroupedSimulation requirementplanningreport
    2. generationsectiongenerationcontentsectionCalltoolGet
    3. Checkcontent
    """

    # toolCalleachsection
    MAX_TOOL_CALLS_PER_SECTION = 5

    # rounds
    MAX_REFLECTION_ROUNDS = 3

    # toolCall
    MAX_TOOL_CALLS_PER_CHAT = 2

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None,
    ):
        """
        InitializeReport Agent

        Args:
            graph_id: ID
            simulation_id: simulation ID
            simulation_requirement: Simulation requirement
            llm_client: LLMoptional
            zep_tools: Zeptooloptional
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()

        # tooldefinition
        self.tools = self._define_tools()

        # logs generate_report Initialize
        self.report_logger: Optional[ReportLogger] = None
        # consolelogs generate_report Initialize
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            f"ReportAgent 초기화 완료: graph_id={graph_id}, simulation_id={simulation_id}"
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """definitiontool"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "깊이 분석하고 싶은 질문 또는 주제",
                    "report_context": "현재 보고서 섹션의 맥락(선택 사항, 더 정확한 하위 질문 생성에 도움)",
                },
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "관련도 정렬에 사용할 검색 질의",
                    "include_expired": "만료/과거 내용을 포함할지 여부(기본 True)",
                },
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "검색 질의 문자열",
                    "limit": "반환할 결과 수(선택 사항, 기본 10)",
                },
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "인터뷰 주제 또는 요구사항 설명(예: 학생들이 특정 사건을 어떻게 보는지)",
                    "max_agents": "최대 인터뷰할 Agent 수(선택 사항, 기본 5, 최대 10)",
                },
            },
        }

    def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """
        toolCall

        Args:
            tool_name: tool
            parameters: toolParameters
            report_context: reportused toInsightForge

        returns:
            tool
        """
        logger.info(f"도구 실행: {tool_name}, 파라미터: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx,
                )
                return result.to_text()

            elif tool_name == "panorama_search":
                #  - Get
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id, query=query, include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                #  - 
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id, query=query, limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                #  - CallOASISAPIGetAgentanswer
                interview_topic = parameters.get(
                    "interview_topic", parameters.get("query", "")
                )
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents,
                )
                return result.to_text()

            # ========== aftertooltool ==========

            elif tool_name == "search_graph":
                #  quick_search
                logger.info("search_graph는 quick_search로 리다이렉트됨")
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id, entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":
                #  insight_forge
                logger.info("get_simulation_context는 insight_forge로 리다이렉트됨")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool(
                    "insight_forge", {"query": query}, report_context
                )

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id, entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return f"알 수 없는 도구: {tool_name}. insight_forge, panorama_search, quick_search 중 하나를 사용해줘"

        except Exception as e:
            logger.error(f"도구 실행 실패: {tool_name}, 오류: {str(e)}")
            return f"도구 실행 실패: {str(e)}"

    # toolused to JSON 
    VALID_TOOL_NAMES = {
        "insight_forge",
        "panorama_search",
        "quick_search",
        "interview_agents",
    }

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from the local model's mixed output formats."""
        tool_calls: List[Dict[str, Any]] = []
        stripped = response.strip()

        def append_if_valid(candidate: Dict[str, Any]) -> None:
            if self._is_valid_tool_call(candidate):
                tool_calls.append(candidate)

        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                append_if_valid(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        if tool_calls:
            return tool_calls

        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                append_if_valid(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        if tool_calls:
            return tool_calls

        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                append_if_valid(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        if tool_calls:
            return tool_calls

        call_pattern = r"<tool_call>\s*([a-zA-Z_][\w]*)\s*\((\{.*?\})\)\s*(?:</tool_call>|$)"
        for match in re.finditer(call_pattern, response, re.DOTALL):
            append_if_valid({
                "name": match.group(1),
                "parameters": self._coerce_tool_parameters(match.group(2)),
            })
        if tool_calls:
            return tool_calls

        function_pattern = r"<tool_call>\s*<function=([a-zA-Z_][\w]*)>(.*?)</function>\s*</tool_call>"
        for match in re.finditer(function_pattern, response, re.DOTALL):
            append_if_valid({
                "name": match.group(1),
                "parameters": self._build_parameter_dict_from_xml(match.group(2)),
            })
        if tool_calls:
            return tool_calls

        token_pattern = r"(?:<tool_call>)?\s*([a-zA-Z_][\w]*)[^\n]*?<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>"
        for match in re.finditer(token_pattern, response, re.DOTALL):
            append_if_valid({
                "name": match.group(1),
                "parameters": self._coerce_tool_parameters(match.group(2)),
            })
        if tool_calls:
            return tool_calls

        loose_pattern = r"(?:<tool_call>\s*)?([a-zA-Z_][\w]*)\s*(\{.*\})"
        match = re.search(loose_pattern, stripped, re.DOTALL)
        if match:
            append_if_valid({
                "name": match.group(1),
                "parameters": self._coerce_tool_parameters(match.group(2)),
            })

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """ JSON whethertoolCall"""
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            if "parameters" not in data or not isinstance(data["parameters"], dict):
                data["parameters"] = self._coerce_tool_parameters(
                    data.get("parameters") or data.get("arguments") or {}
                )
            else:
                data["parameters"] = self._coerce_tool_parameters(data["parameters"])
            return True
        return False

    def _coerce_tool_parameters(self, parameters: Any) -> Dict[str, Any]:
        """Normalize model-produced tool arguments into a dict."""
        if parameters is None:
            return {}
        if isinstance(parameters, dict):
            return parameters
        if isinstance(parameters, str):
            stripped = parameters.strip()
            if not stripped:
                return {}
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, dict):
                    return decoded
            except json.JSONDecodeError:
                pass
            return {"query": stripped}
        return {}

    def _build_parameter_dict_from_xml(self, body: str) -> Dict[str, Any]:
        """Parse <parameter=name>value</parameter> blocks."""
        params: Dict[str, Any] = {}
        for match in re.finditer(
            r"<parameter=([a-zA-Z_][\w-]*)>\s*(.*?)\s*</parameter>",
            body,
            re.DOTALL,
        ):
            key = match.group(1)
            value = match.group(2).strip()
            if value.lower() in {"true", "false"}:
                params[key] = value.lower() == "true"
                continue
            try:
                params[key] = json.loads(value)
            except json.JSONDecodeError:
                params[key] = value
        return params

    def _looks_like_tool_attempt(self, response: str) -> bool:
        lowered = response.lower()
        return any(
            marker in lowered
            for marker in (
                "<tool_call>",
                "<function=",
                "tool_call_argument_begin",
                "insight_forge",
                "panorama_search",
                "quick_search",
                "interview_agents",
            )
        )

    def _get_tools_description(self) -> str:
        """generationtool"""
        desc_parts = ["사용 가능한 도구:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join(
                [f"{k}: {v}" for k, v in tool["parameters"].items()]
            )
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  파라미터: {params_desc}")
        return "\n".join(desc_parts)

    def _extract_used_quotes(self, content: str) -> List[str]:
        quotes = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('>'):
                quote = stripped.lstrip('> ').strip()
                if quote:
                    quotes.append(quote)
        return quotes

    def _build_previous_section_context(self, previous_sections: List[str]) -> str:
        if not previous_sections:
            return "(첫 번째 섹션임)"

        contexts = []
        for sec in previous_sections:
            lines = sec.splitlines()
            title = lines[0].lstrip('# ').strip() if lines else '이전 섹션'
            body = "\n".join(lines[1:]).strip()
            summary = body[:1200] + "..." if len(body) > 1200 else body
            contexts.append(f"- {title}:\n{summary}")
        return "\n\n".join(contexts)

    def _build_used_quote_context(self, previous_sections: List[str]) -> str:
        quotes: List[str] = []
        seen = set()
        for sec in previous_sections:
            for quote in self._extract_used_quotes(sec):
                if quote not in seen:
                    seen.add(quote)
                    quotes.append(quote)
        if not quotes:
            return "(아직 사용한 핵심 인용이 없음)"
        return "\n".join(f'- {quote}' for quote in quotes[:12])

    def _revise_section_if_needed(
        self,
        section_title: str,
        content: str,
        previous_sections: List[str],
        messages: List[Dict[str, str]],
    ) -> str:
        previous_quotes = set(self._extract_used_quotes("\n".join(previous_sections)))
        if not previous_quotes:
            return content

        current_quotes = self._extract_used_quotes(content)
        overlap = [quote for quote in current_quotes if quote in previous_quotes]
        if len(overlap) < 2:
            return content

        revision_messages = messages + [
            {"role": "assistant", "content": f"Final Answer: {content}"},
            {
                "role": "user",
                "content": (
                    "[품질 재작성] 방금 쓴 본문은 이전 섹션과 같은 인용을 반복했다.\n"
                    f"겹치는 인용: {overlap[:5]}\n"
                    "이미 수집된 도구 결과 안에서 다른 사실과 인용을 골라 같은 섹션을 다시 써라. "
                    "겹치는 인용은 다시 쓰지 말고, 한국어 본문만 Final Answer: 형식으로 출력하라."
                ),
            },
        ]


        revised = self.llm.chat(messages=revision_messages, temperature=0.4, max_tokens=4096)
        if not revised:
            return content

        revised_content = (
            revised.split("Final Answer:")[-1].strip()
            if "Final Answer:" in revised
            else revised.strip()
        )
        revised_quotes = self._extract_used_quotes(revised_content)
        revised_overlap = [quote for quote in revised_quotes if quote in previous_quotes]
        return revised_content if len(revised_overlap) < len(overlap) else content

    def plan_outline(
        self, progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        planningreport

        LLMgroupedSimulation requirementplanningreport

        Args:
            progress_callback: Progress callback

        returns:
            ReportOutline: report
        """
        logger.info("보고서 개요 구성을 시작함...")

        if progress_callback:
            progress_callback("planning", 0, "시뮬레이션 요구사항을 분석 중이야...")

        # Get
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id, simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "보고서 개요를 생성 중이야...")

        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
            total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
            entity_types=list(
                context.get("graph_statistics", {}).get("entity_types", {}).keys()
            ),
            relation_types=json.dumps(
                context.get("graph_statistics", {}).get("relation_types", {}),
                ensure_ascii=False,
            ),
            total_entities=context.get("total_entities", 0),
            representative_entities_json=json.dumps(
                context.get("entities", [])[:15], ensure_ascii=False, indent=2
            ),
            related_facts_json=json.dumps(
                context.get("related_facts", [])[:30], ensure_ascii=False, indent=2
            ),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            if progress_callback:
                progress_callback("planning", 80, "개요 구조를 정리 중이야...")

            # 
            sections = []
            for section_data in response.get("sections", []):
                sections.append(
                    ReportSection(
                        title=section_data.get("title", ""),
                        content="",
                        description=section_data.get("description", ""),
                    )
                )

            outline = ReportOutline(
                title=response.get("title", "시뮬레이션 분석 보고서"),
                summary=response.get("summary", ""),
                sections=sections,
            )

            if progress_callback:
                progress_callback("planning", 100, "개요 구성이 완료됐어")

            logger.info(f"개요 구성이 완료됨: 총 {len(sections)}개 섹션")
            return outline

        except Exception as e:
            logger.error(f"개요 구성 실패: {str(e)}")
            # returnsdefault3sectionfallback
            return ReportOutline(
                title="미래 예측 보고서",
                summary="시뮬레이션 기반 미래 흐름과 리스크 분석",
                sections=[
                    ReportSection(title="예측 시나리오와 핵심 발견", description="핵심 사건 전개와 초기 충격을 요약"),
                    ReportSection(title="집단 행동 예측 분석", description="주요 집단의 반응과 프레임 분화를 분석"),
                    ReportSection(title="향후 흐름 전망과 리스크 시사점", description="전환 포인트와 시스템 리스크를 정리"),
                ],
            )

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0,
    ) -> str:
        """
        ReACTgenerationsinglesectioncontent

        ReACT
        1. Thought- grouped
        2. Action- CalltoolGet
        3. Observation- groupedtoolreturns
        4. 
        5. Final Answeranswer- generationsectioncontent

        Args:
            section: generationsection
            outline: 
            previous_sections: sectioncontentused to
            progress_callback: Progress callback
            section_index: sectionused tologs

        returns:
            sectioncontentMarkdown
        """
        logger.info(f"ReACT로 섹션 생성 시작: {section.title}")

        # sectionStartlogs
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            section_description=section.description or "이 섹션의 고유한 질문에 답하라",
            tools_description=self._get_tools_description(),
        )

        previous_section_context = self._build_previous_section_context(previous_sections)
        used_quotes = self._build_used_quote_context(previous_sections)

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_section_context=previous_section_context,
            used_quotes=used_quotes,
            section_title=section.title,
        )


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ReACT
        tool_calls_count = 0
        max_iterations = 8  # rounds
        min_tool_calls = 1  # Require at least one successful evidence fetch
        conflict_retries = 0  # toolCallFinal Answer
        used_tools = set()  # Calltool
        all_tools = {
            "insight_forge",
            "panorama_search",
            "quick_search",
            "interview_agents",
        }

        # reportused toInsightForgegeneration
        report_context = (
            f"섹션 제목: {section.title}\n시뮬레이션 요구사항: {self.simulation_requirement}"
        )

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"심층 검색과 작성 진행 중 ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})",
                )

            # CallLLM
            response = self.llm.chat(
                messages=messages, temperature=0.5, max_tokens=4096
            )

            # Check LLM returnswhether NoneAPI content
            if response is None:
                logger.warning(
                    f"섹션 {section.title} {iteration + 1}회차 반복: LLM이 None을 반환함"
                )
                # hasadd
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(응답이 비어 있음)"})
                    messages.append({"role": "user", "content": "내용 생성을 계속해라."})
                    continue
                # afterreturns None
                break

            logger.debug(f"LLM 응답: {response[:200]}...")

            # 
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── LLM outputtoolCall Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"섹션 {section.title} {iteration + 1}회차: "
                    f"LLM이 도구 호출과 Final Answer를 동시에 출력함 ({conflict_retries}번째 충돌)"
                )

                if conflict_retries <= 2:
                    #  LLM 
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[형식 오류] 한 번의 응답에 도구 호출과 Final Answer를 동시에 넣으면 안 된다.\n"
                                "각 응답에서는 아래 둘 중 하나만 수행하라.\n"
                                "- 도구 하나를 호출한다 (<tool_call> 블록만 출력, Final Answer 금지)\n"
                                "- 최종 본문을 출력한다 ('Final Answer:'로 시작, <tool_call> 금지)\n"
                                "다시 응답하고 둘 중 하나만 수행하라."
                            ),
                        }
                    )
                    continue
                else:
                    # toolCall
                    logger.warning(
                        f"섹션 {section.title}: {conflict_retries}회 연속 충돌, "
                        "첫 번째 도구 호출만 남기고 강제로 계속 진행함"
                    )
                    first_tool_end = response.find("</tool_call>")
                    if first_tool_end != -1:
                        response = response[: first_tool_end + len("</tool_call>")]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            #  LLM logs
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer,
                )

            # ── 1LLM output Final Answer ──
            if has_final_answer:
                # toolCalltool
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f"(아직 사용하지 않은 도구: {', '.join(unused_tools)})"
                        if unused_tools
                        else ""
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                                tool_calls_count=tool_calls_count,
                                min_tool_calls=min_tool_calls,
                                unused_hint=unused_hint,
                            ),
                        }
                    )
                    continue

                # 
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(
                    f"섹션 {section.title} 생성 완료 (도구 호출: {tool_calls_count}회)"
                )

                final_answer = self._revise_section_if_needed(
                    section.title, final_answer, previous_sections, messages
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count,
                    )
                return final_answer

            # ── 2LLM Calltool ──
            if has_tool_calls:
                # tool → output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_TOOL_LIMIT_MSG.format(
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            ),
                        }
                    )
                    continue

                # toolCall
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(
                        f"LLM이 {len(tool_calls)}개 도구를 요청했지만 첫 번째만 실행함: {call['name']}"
                    )

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1,
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context,
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1,
                    )

                tool_calls_count += 1
                used_tools.add(call["name"])

                # buildtool
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(
                        unused_list="".join(unused_tools)
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": REACT_OBSERVATION_TEMPLATE.format(
                            tool_name=call["name"],
                            result=result,
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            used_tools_str=", ".join(used_tools),
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # ── 3hastoolCallhas Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if self._looks_like_tool_attempt(response) and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f"(아직 사용하지 않은 도구: {', '.join(unused_tools)})"
                    if unused_tools
                    else ""
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[도구 형식 오류] 도구 호출 의도는 보였지만 파싱되지 않았다. "
                            "반드시 아래 형식 중 하나로만 다시 출력하라.\n"
                            "1) <tool_call>{\"name\": \"insight_forge\", \"parameters\": {\"query\": \"...\"}}</tool_call>\n"
                            "2) <tool_call>{\"name\": \"panorama_search\", \"parameters\": {\"query\": \"...\", \"include_expired\": false}}</tool_call>\n"
                            "설명문이나 Final Answer를 섞지 말고 도구 호출 블록만 출력하라."
                            + unused_hint
                        ),
                    }
                )
                continue

            if tool_calls_count < min_tool_calls:
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f"(아직 사용하지 않은 도구: {', '.join(unused_tools)})"
                    if unused_tools
                    else ""
                )

                messages.append(
                    {
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            logger.info(
                f"섹션 {section.title}에서 'Final Answer:' 접두어가 없어도 현재 출력을 최종 본문으로 채택함 (도구 호출: {tool_calls_count}회)"
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count,
                )
            return final_answer

        # generationcontent
        logger.warning(f"섹션 {section.title}가 최대 반복 횟수에 도달해 강제 마무리함")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(messages=messages, temperature=0.5, max_tokens=4096)

        # Check LLM returnswhether None
        if response is None:
            logger.error(
                f"섹션 {section.title} 강제 마무리 중 LLM이 None을 반환해 기본 오류 문구를 사용함"
            )
            final_answer = f"(이 섹션 생성 실패: LLM이 빈 응답을 반환함. 잠시 후 다시 시도해줘)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # sectioncontentgenerationcompletelogs
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None,
    ) -> Report:
        """
        generationreportgroupedsectionreal-timeoutput

        eachsectiongenerationcompleteafterSaveWaitingreportcomplete
        
        reports/{report_id}/
            meta.json       - report
            outline.json    - report
            progress.json   - generationprogress
            section_01.md   - 1section
            section_02.md   - 2section
            ...
            full_report.md  - report

        Args:
            progress_callback: Progress callback (stage, progress, message)
            report_id: reportIDoptional, generation

        returns:
            Report: report
        """
        import uuid

        # has report_idgeneration
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )

        # Donesectionlistused toprogress
        completed_section_titles = []

        try:
            # InitializecreatereportSavestate
            ReportManager._ensure_report_folder(report_id)

            # Initializelogsstructuredlogs agent_log.jsonl
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
            )

            # Initializeconsolelogsconsole_log.txt
            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id,
                "pending",
                0,
                "보고서를 초기화하는 중...",
                completed_sections=[],
            )
            ReportManager.save_report(report)

            # 1: planning
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id,
                "planning",
                5,
                "보고서 개요 구성을 시작하는 중...",
                completed_sections=[],
            )

            # planningStartlogs
            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback("planning", 0, "보고서 개요 구성을 시작하는 중...")

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: progress_callback(
                    stage, prog // 5, msg
                )
                if progress_callback
                else None
            )
            report.outline = outline

            # planningcompletelogs
            self.report_logger.log_planning_complete(outline.to_dict())

            # Save
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "planning",
                15,
                f"개요 구성이 완료됐어. 총 {len(outline.sections)}개 섹션이야",
                completed_sections=[],
            )
            ReportManager.save_report(report)

            logger.info(f"개요 파일 저장 완료: {report_id}/outline.json")

            # 2: sectiongenerationgroupedsectionSave
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # Savecontentused to

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # updateprogress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress,
                    f"섹션 생성 중: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles,
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"섹션 생성 중: {section.title} ({section_num}/{total_sections})",
                    )

                # generationsectioncontent
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg: progress_callback(
                        stage, base_progress + int(prog * 0.7 / total_sections), msg
                    )
                    if progress_callback
                    else None,
                    section_index=section_num,
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Savesection
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # sectioncompletelogs
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip(),
                    )

                logger.info(f"섹션 저장 완료: {report_id}/section_{section_num:02d}.md")

                # updateprogress
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress + int(70 / total_sections),
                    f"섹션 {section.title} 완료",
                    current_section=None,
                    completed_sections=completed_section_titles,
                )

            # 3: report
            if progress_callback:
                progress_callback("generating", 95, "최종 보고서를 조립하는 중이야...")

            ReportManager.update_progress(
                report_id,
                "generating",
                95,
                "최종 보고서를 조립하는 중이야...",
                completed_sections=completed_section_titles,
            )

            # ReportManagerreport
            report.markdown_content = ReportManager.assemble_full_report(
                report_id, outline
            )
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # 
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # reportcompletelogs
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections, total_time_seconds=total_time_seconds
                )

            # Savereport
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id,
                "completed",
                100,
                "보고서 생성이 완료됐어",
                completed_sections=completed_section_titles,
            )

            if progress_callback:
                progress_callback("completed", 100, "보고서 생성이 완료됐어")

            logger.info(f"보고서 생성 완료: {report_id}")

            # consolelogs
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"보고서 생성 실패: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # logs
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Savestate
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id,
                    "failed",
                    -1,
                    f"보고서 생성 실패: {str(e)}",
                    completed_sections=completed_section_titles,
                )
            except Exception:
                pass  # Save

            # consolelogs
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self, message: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with Report Agent

        AgentCalltoolanswer

        Args:
            message: user message
            chat_history: chat history

        returns:
            {
                "response": "Agent 응답",
                "tool_calls": [list of called tools],
                "sources": [information sources]
            }
        """
        logger.info(f"Report Agent 대화: {message[:50]}...")

        chat_history = chat_history or []

        # Getgenerationreportcontent
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # report
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [보고서 내용이 잘림] ..."
        except Exception as e:
            logger.warning(f"보고서 내용을 가져오지 못함: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(아직 보고서 없음)",
            tools_description=self._get_tools_description(),
        )

        # build
        messages = [{"role": "system", "content": system_prompt}]

        # add
        for h in chat_history[-10:]:  # 
            messages.append(h)

        # adduser message
        messages.append({"role": "user", "content": message})

        # ReACT
        tool_calls_made = []
        max_iterations = 2  # rounds

        for iteration in range(max_iterations):
            response = self.llm.chat(messages=messages, temperature=0.5)

            # toolCall
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # hastoolCallreturns
                clean_response = re.sub(
                    r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL
                )
                clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [
                        tc.get("parameters", {}).get("query", "")
                        for tc in tool_calls_made
                    ],
                }

            # toolCall
            tool_results = []
            for call in tool_calls[:1]:  # rounds1toolCall
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(
                    {
                        "tool": call["name"],
                        "result": result[:1500],  # 
                    }
                )
                tool_calls_made.append(call)

            # add
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join(
                [f"[{r['tool']} 결과]\n{r['result']}" for r in tool_results]
            )
            messages.append(
                {"role": "user", "content": observation + CHAT_OBSERVATION_SUFFIX}
            )

        # Get
        final_response = self.llm.chat(messages=messages, temperature=0.5)

        # 
        clean_response = re.sub(
            r"<tool_call>.*?</tool_call>", "", final_response, flags=re.DOTALL
        )
        clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [
                tc.get("parameters", {}).get("query", "") for tc in tool_calls_made
            ],
        }


class ReportManager:
    """
    report

    report

    groupedsectionoutput
    reports/
      {report_id}/
        meta.json          - reportstate
        outline.json       - report
        progress.json      - generationprogress
        section_01.md      - 1section
        section_02.md      - 2section
        ...
        full_report.md     - report
    """

    # report
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "reports")

    @classmethod
    def _ensure_reports_dir(cls):
        """report"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Getreportpath"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """reportreturnspath"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Getreportpath"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """GetreportMarkdown filepath"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Getpath"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Getprogresspath"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """GetsectionMarkdown filepath"""
        return os.path.join(
            cls._get_report_folder(report_id), f"section_{section_index:02d}.md"
        )

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get Agent logspath"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Getconsolelogspath"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Getconsolelogscontent

        reportgenerationconsoleoutputlogsINFOWARNING
         agent_log.jsonl structuredlogsdifferent

        Args:
            report_id: reportID
            from_line: read starting from lineused toGet0 Start

        returns:
            {
                "logs": [logslist],
                "total_lines": ,
                "from_line": ,
                "has_more": whetherhaslogs
            }
        """
        log_path = cls._get_console_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # logs
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # 
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Getcompleteconsolelogsone-shotGetall

        Args:
            report_id: reportID

        returns:
            logslist
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent logscontent

        Args:
            report_id: reportID
            from_line: read starting from lineused toGet0 Start

        returns:
            {
                "logs": [logslist],
                "total_lines": ,
                "from_line": ,
                "has_more": whetherhaslogs
            }
        """
        log_path = cls._get_agent_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # 
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # 
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Getcomplete Agent logsused toone-shotGetall

        Args:
            report_id: reportID

        returns:
            logslist
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save report

        planningcompleteafterCall
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), "w", encoding="utf-8") as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"개요 저장 완료: {report_id}")

    @classmethod
    def save_section(
        cls, report_id: str, section_index: int, section: ReportSection
    ) -> str:
        """
        Savesinglesection

        eachsectiongenerationcompleteafterCallgroupedsectionoutput

        Args:
            report_id: reportID
            section_index: section1Start
            section: section

        returns:
            Savepath
        """
        cls._ensure_report_folder(report_id)

        # buildsectionMarkdowncontent - 
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save files
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"섹션 저장 완료: {report_id}/{file_suffix}")
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        sectioncontent

        1. contentsectionMarkdown
        2. has ### 

        Args:
            content: content
            section_title: section

        returns:
            aftercontent
        """
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split("\n")
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # CheckwhetherMarkdown
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Checkwhethersection5
                if i < 5:
                    if title_text == section_title or title_text.replace(
                        " ", ""
                    ) == section_title.replace(" ", ""):
                        skip_next_empty = True
                        continue

                # has#, ##, ###, ####
                # sectionaddcontenthas
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # add
                continue

            # 
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # 
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)

        # grouped
        while cleaned_lines and cleaned_lines[0].strip() in ["---", "***", "___"]:
            cleaned_lines.pop(0)
            # groupedafter
            while cleaned_lines and cleaned_lines[0].strip() == "":
                cleaned_lines.pop(0)

        return "\n".join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None,
    ) -> None:
        """
        updatereportgenerationprogress

        frontendprogress.jsonGetreal-timeprogress
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat(),
        }

        with open(cls._get_progress_path(report_id), "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Getreportgenerationprogress"""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Getgenerationsectionlist

        returnshasSavesection
        """
        folder = cls._get_report_folder(report_id)

        if not os.path.exists(folder):
            return []

        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("section_") and filename.endswith(".md"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # section
                parts = filename.replace(".md", "").split("_")
                section_index = int(parts[1])

                sections.append(
                    {
                        "filename": filename,
                        "section_index": section_index,
                        "content": content,
                    }
                )

        return sections

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        report

        Savesectionreport
        """
        folder = cls._get_report_folder(report_id)

        # buildreport
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # hassection
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # afterreport
        md_content = cls._post_process_report(md_content, outline)

        # Savereport
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"전체 보고서 조립 완료: {report_id}")
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        afterreportcontent

        1. 
        2. report(#)section(##)Other(###, ####)
        3. grouped

        Args:
            content: reportcontent
            outline: report

        returns:
            aftercontent
        """
        import re

        lines = content.split("\n")
        processed_lines = []
        prev_was_heading = False

        # hassection
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Checkwhether
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Checkwhether5content
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r"^(#{1,6})\s+(.+)$", prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # after
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue

                # 
                # - # (level=1) report
                # - ## (level=2) section
                # - ###  (level>=3) 

                if level == 1:
                    if title == outline.title:
                        # report
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # section###
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # section
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # section
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### 
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == "---" and prev_was_heading:
                # aftergrouped
                i += 1
                continue

            elif stripped == "" and prev_was_heading:
                # after
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # 2
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save reportreport"""
        cls._ensure_report_folder(report.report_id)

        # SaveJSON
        with open(cls._get_report_path(report.report_id), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Save
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # SaveMarkdownreport
        if report.markdown_content:
            with open(
                cls._get_report_markdown_path(report.report_id), "w", encoding="utf-8"
            ) as f:
                f.write(report.markdown_content)

        logger.info(f"보고서 저장 완료: {report.report_id}")

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Getreport"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Checkreports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Report
        outline = None
        if data.get("outline"):
            outline_data = data["outline"]
            sections = []
            for s in outline_data.get("sections", []):
                sections.append(
                    ReportSection(
                        title=s["title"],
                        content=s.get("content", ""),
                        description=s.get("description", ""),
                    )
                )
            outline = ReportOutline(
                title=outline_data["title"],
                summary=outline_data["summary"],
                sections=sections,
            )

        # markdown_contentfull_report.md
        markdown_content = data.get("markdown_content", "")
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

        return Report(
            report_id=data["report_id"],
            simulation_id=data["simulation_id"],
            graph_id=data["graph_id"],
            simulation_requirement=data["simulation_requirement"],
            status=ReportStatus(data["status"]),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error"),
        )

    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get report by simulation ID"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # JSON
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report

        return None

    @classmethod
    def list_reports(
        cls, simulation_id: Optional[str] = None, limit: int = 50
    ) -> List[Report]:
        """report"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # JSON
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # create
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # 
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"보고서 폴더 삭제 완료: {report_id}")
            return True

        # 
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")

        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True

        return deleted











