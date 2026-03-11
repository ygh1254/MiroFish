"""
Report API routes
Provides endpoints for report generation, retrieval, and conversation
"""

import os
import traceback
import threading
from flask import request, jsonify, send_file

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.report")


# ============== Report generation API ==============


@report_bp.route("/generate", methods=["POST"])
def generate_report():
    """
    Generate a simulation analysis report (async)

    This is a long-running operation; the API returns task_id immediately,
     GET /api/report/generate/status check progress

    Request (JSON)
        {
            "simulation_id": "sim_xxxx",    // required, simulation ID
            "force_regenerate": false        // optional, force regenerate
        }

    returns
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "보고서 생성 작업을 시작했어"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify({"success": False, "error": "simulation_id를 입력해줘"}), 400

        force_regenerate = data.get("force_regenerate", False)

        # Get simulation information
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"simulation이 존재하지 않아: {simulation_id}"}
            ), 404

        # Check whether a report already exists
        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "report_id": existing_report.report_id,
                            "status": "completed",
                            "message": "보고서가 이미 존재해",
                            "already_generated": True,
                        },
                    }
                )

        # Get project information
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {"success": False, "error": f"project가 존재하지 않아: {state.project_id}"}
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify(
                {"success": False, "error": "graph_id가 없어. 그래프 생성이 완료됐는지 확인해줘"}
            ), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({"success": False, "error": "simulation requirement가 없어"}), 400

        # Pre-generate report_id so it can be returned to the frontend immediately
        import uuid

        report_id = f"report_{uuid.uuid4().hex[:12]}"

        # Create async task
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id,
            },
        )

        # Define background task
        def run_generate():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Report Agent 초기화 중...",
                )

                # Create Report Agent
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                )

                # Progress callback
                def progress_callback(stage, progress, message):
                    task_manager.update_task(
                        task_id, progress=progress, message=f"[{stage}] {message}"
                    )

                # Generate report with the pre-generated report_id
                report = agent.generate_report(
                    progress_callback=progress_callback, report_id=report_id
                )

                # Save report
                ReportManager.save_report(report)

                if report.status == ReportStatus.COMPLETED:
                    task_manager.complete_task(
                        task_id,
                        result={
                            "report_id": report.report_id,
                            "simulation_id": simulation_id,
                            "status": "completed",
                        },
                    )
                else:
                    task_manager.fail_task(
                        task_id, report.error or "보고서 생성에 실패했어"
                    )

            except Exception as e:
                logger.error(f"보고서 생성 실패: {str(e)}")
                task_manager.fail_task(task_id, str(e))

        # Start background thread
        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "report_id": report_id,
                    "task_id": task_id,
                    "status": "generating",
                    "message": "보고서 생성 작업을 시작했어. /api/report/generate/status 로 진행 상황을 확인해줘",
                    "already_generated": False,
                },
            }
        )

    except Exception as e:
        logger.error(f"보고서 생성 작업 시작 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/generate/status", methods=["POST"])
def get_generate_status():
    """
    Queryreportgenerationprogress

    Request (JSON)
        {
            "task_id": "task_xxxx",         // optional, generation returnstask_id
            "simulation_id": "sim_xxxx"     // optional, simulation ID
        }

    returns
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|failed",
                "progress": 45,
                "message": "..."
            }
        }
    """
    try:
        data = request.get_json() or {}

        task_id = data.get("task_id")
        simulation_id = data.get("simulation_id")

        # If simulation_id is provided, first check whether a completed report already exists
        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "report_id": existing_report.report_id,
                            "status": "completed",
                            "progress": 100,
                            "message": "보고서 생성이 완료됐어",
                            "already_completed": True,
                        },
                    }
                )

        if not task_id:
            return jsonify(
                {"success": False, "error": "task_id 또는 simulation_id를 입력해줘"}
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify({"success": False, "error": f"task가 존재하지 않아: {task_id}"}), 404

        return jsonify({"success": True, "data": task.to_dict()})

    except Exception as e:
        logger.error(f"작업 상태 조회 실패: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Report retrieval API ==============


@report_bp.route("/<report_id>", methods=["GET"])
def get_report(report_id: str):
    """
    Get report details

    returns
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "simulation_id": "sim_xxxx",
                "status": "completed",
                "outline": {...},
                "markdown_content": "...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify({"success": False, "error": f"report가 존재하지 않아: {report_id}"}), 404

        return jsonify({"success": True, "data": report.to_dict()})

    except Exception as e:
        logger.error(f"보고서 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/by-simulation/<simulation_id>", methods=["GET"])
def get_report_by_simulation(simulation_id: str):
    """
    Get report by simulation ID

    returns
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                ...
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        if not report:
            return jsonify(
                {
                    "success": False,
                    "error": f"이 simulation에 대한 report가 아직 없어: {simulation_id}",
                    "has_report": False,
                }
            ), 404

        return jsonify({"success": True, "data": report.to_dict(), "has_report": True})

    except Exception as e:
        logger.error(f"보고서 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/list", methods=["GET"])
def list_reports():
    """
    List all reports

    QueryParameters
        simulation_id: Filter by simulation IDoptional
        limit: returnslimitdefault50

    returns
        {
            "success": true,
            "data": [...],
            "count": 10
        }
    """
    try:
        simulation_id = request.args.get("simulation_id")
        limit = request.args.get("limit", 50, type=int)

        reports = ReportManager.list_reports(simulation_id=simulation_id, limit=limit)

        return jsonify(
            {
                "success": True,
                "data": [r.to_dict() for r in reports],
                "count": len(reports),
            }
        )

    except Exception as e:
        logger.error(f"보고서 목록 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/download", methods=["GET"])
def download_report(report_id: str):
    """
    Download report (Markdown)

    returnsMarkdown file
    """
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify({"success": False, "error": f"report가 존재하지 않아: {report_id}"}), 404

        md_path = ReportManager._get_report_markdown_path(report_id)

        if not os.path.exists(md_path):
            # If the Markdown file is missing, generate a temporary file
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name

            return send_file(
                temp_path, as_attachment=True, download_name=f"{report_id}.md"
            )

        return send_file(md_path, as_attachment=True, download_name=f"{report_id}.md")

    except Exception as e:
        logger.error(f"보고서 다운로드 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>", methods=["DELETE"])
def delete_report(report_id: str):
    """Delete report"""
    try:
        success = ReportManager.delete_report(report_id)

        if not success:
            return jsonify({"success": False, "error": f"report가 존재하지 않아: {report_id}"}), 404

        return jsonify({"success": True, "message": f"report 삭제 완료: {report_id}"})

    except Exception as e:
        logger.error(f"보고서 삭제 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Report Agentconversation API ==============


@report_bp.route("/chat", methods=["POST"])
def chat_with_report_agent():
    """
    Chat with Report Agent

    Report AgentCalltoolanswer

    Request (JSON)
        {
            "simulation_id": "sim_xxxx",        // required, simulation ID
            "message": "흐름이 어떻게 전개되는지 설명해줘",    // required, user message
            "chat_history": [                   // optional, chat history
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    returns
        {
            "success": true,
            "data": {
                "response": "Agent 응답...",
                "tool_calls": [list of called tools],
                "sources": [information sources]
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        message = data.get("message")
        chat_history = data.get("chat_history", [])

        if not simulation_id:
            return jsonify({"success": False, "error": "simulation_id를 입력해줘"}), 400

        if not message:
            return jsonify({"success": False, "error": "message를 입력해줘"}), 400

        # Get
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"simulation이 존재하지 않아: {simulation_id}"}
            ), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {"success": False, "error": f"project가 존재하지 않아: {state.project_id}"}
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "graph_id가 없어"}), 400

        simulation_requirement = project.simulation_requirement or ""

        # Create the Agent and run the conversation
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
        )

        result = agent.chat(message=message, chat_history=chat_history)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"대화 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== report progress and section APIs ==============


@report_bp.route("/<report_id>/progress", methods=["GET"])
def get_report_progress(report_id: str):
    """
    Get report generation progress (real time)

    returns
        {
            "success": true,
            "data": {
                "status": "generating",
                "progress": 45,
                "message": "섹션 생성 중: 핵심 발견",
                "current_section": "핵심 발견",
                "completed_sections": ["실행 요약", "시뮬레이션 배경"],
                "updated_at": "2025-12-09T..."
            }
        }
    """
    try:
        progress = ReportManager.get_progress(report_id)

        if not progress:
            return jsonify(
                {"success": False, "error": f"report가 없거나 progress 정보를 읽을 수 없어: {report_id}"}
            ), 404

        return jsonify({"success": True, "data": progress})

    except Exception as e:
        logger.error(f"보고서 진행률 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/sections", methods=["GET"])
def get_report_sections(report_id: str):
    """
    Get generated section list (grouped by section)

    The frontend can poll this API for generated section content without waiting for the whole report to finish

    returns
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "sections": [
                    {
                        "filename": "section_01.md",
                        "section_index": 1,
                        "content": "## Executive Summary\\n\\n..."
                    },
                    ...
                ],
                "total_sections": 3,
                "is_complete": false
            }
        }
    """
    try:
        sections = ReportManager.get_generated_sections(report_id)

        # Getreportstate
        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED

        return jsonify(
            {
                "success": True,
                "data": {
                    "report_id": report_id,
                    "sections": sections,
                    "total_sections": len(sections),
                    "is_complete": is_complete,
                },
            }
        )

    except Exception as e:
        logger.error(f"섹션 목록 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/section/<int:section_index>", methods=["GET"])
def get_single_section(report_id: str, section_index: int):
    """
    Get a single section's content

    returns
        {
            "success": true,
            "data": {
                "filename": "section_01.md",
                "content": "## Executive Summary\\n\\n..."
            }
        }
    """
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)

        if not os.path.exists(section_path):
            return jsonify(
                {
                    "success": False,
                    "error": f"section이 존재하지 않아: section_{section_index:02d}.md",
                }
            ), 404

        with open(section_path, "r", encoding="utf-8") as f:
            content = f.read()

        return jsonify(
            {
                "success": True,
                "data": {
                    "filename": f"section_{section_index:02d}.md",
                    "section_index": section_index,
                    "content": content,
                },
            }
        )

    except Exception as e:
        logger.error(f"섹션 내용 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== report status check API ==============


@report_bp.route("/check/<simulation_id>", methods=["GET"])
def check_report_status(simulation_id: str):
    """
    Checkwhetherhasreportandreportstate

    used tofrontenddeterminewhetherunlockInterviewfeature

    returns
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "has_report": true,
                "report_status": "completed",
                "report_id": "report_xxxx",
                "interview_unlocked": true
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        has_report = report is not None
        report_status = report.status.value if report else None
        report_id = report.report_id if report else None

        # onlyreportcompleteafteronly thenunlockinterview
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "has_report": has_report,
                    "report_status": report_status,
                    "report_id": report_id,
                    "interview_unlocked": interview_unlocked,
                },
            }
        )

    except Exception as e:
        logger.error(f"보고서 상태 확인 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Agent log API ==============


@report_bp.route("/<report_id>/agent-log", methods=["GET"])
def get_agent_log(report_id: str):
    """
    Fetch detailed execution logs for Report Agent

    Fetch every step from the report generation process in real time, including:
    - reportStartplanningStart/complete
    - eachsectionStarttoolCallLLMcomplete
    - reportcomplete

    QueryParameters
        from_line: read starting from lineoptional, default0used toGet

    returns
        {
            "success": true,
            "data": {
                "logs": [
                    {
                        "timestamp": "2025-12-13T...",
                        "elapsed_seconds": 12.5,
                        "report_id": "report_xxxx",
                        "action": "tool_call",
                        "stage": "generating",
                        "section_title": "실행 요약",
                        "section_index": 1,
                        "details": {
                            "tool_name": "insight_forge",
                            "parameters": {...},
                            ...
                        }
                    },
                    ...
                ],
                "total_lines": 25,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get("from_line", 0, type=int)

        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)

        return jsonify({"success": True, "data": log_data})

    except Exception as e:
        logger.error(f"Agent 로그 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/agent-log/stream", methods=["GET"])
def stream_agent_log(report_id: str):
    """
    Get the full Agent log in one shot

    returns
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 25
            }
        }
    """
    try:
        logs = ReportManager.get_agent_log_stream(report_id)

        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})

    except Exception as e:
        logger.error(f"Agent 로그 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== console log API ==============


@report_bp.route("/<report_id>/console-log", methods=["GET"])
def get_console_log(report_id: str):
    """
    Fetch console logs for Report Agent

    real-timeGetreportgenerationconsoleoutputINFOWARNING
    this differs from agent-log APIreturnsstructured JSON logsdifferent
    plain-text formatconsolelogs

    QueryParameters
        from_line: read starting from lineoptional, default0used toGet

    returns
        {
            "success": true,
            "data": {
                "logs": [
                    "[19:46:14] INFO: 검색 완료: 관련 사실 15건 발견",
                    "[19:46:14] INFO: 그래프 검색: graph_id=xxx, query=...",
                    ...
                ],
                "total_lines": 100,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get("from_line", 0, type=int)

        log_data = ReportManager.get_console_log(report_id, from_line=from_line)

        return jsonify({"success": True, "data": log_data})

    except Exception as e:
        logger.error(f"콘솔 로그 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/console-log/stream", methods=["GET"])
def stream_console_log(report_id: str):
    """
    Getcompleteconsolelogsone-shotGetall

    returns
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 100
            }
        }
    """
    try:
        logs = ReportManager.get_console_log_stream(report_id)

        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})

    except Exception as e:
        logger.error(f"콘솔 로그 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== toolCallAPI==============


@report_bp.route("/tools/search", methods=["POST"])
def search_graph_tool():
    """
    Graph search tool API (for debugging)

    Request (JSON)
        {
            "graph_id": "mirofish_xxxx",
            "query": "검색 질의",
            "limit": 10
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")
        query = data.get("query")
        limit = data.get("limit", 10)

        if not graph_id or not query:
            return jsonify({"success": False, "error": "graph_id와 query를 입력해줘"}), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)

        return jsonify({"success": True, "data": result.to_dict()})

    except Exception as e:
        logger.error(f"그래프 검색 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/tools/statistics", methods=["POST"])
def get_graph_statistics_tool():
    """
    Graph statistics tool API (for debugging)

    Request (JSON)
        {
            "graph_id": "mirofish_xxxx"
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")

        if not graph_id:
            return jsonify({"success": False, "error": "graph_id를 입력해줘"}), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.get_graph_statistics(graph_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"그래프 통계 조회 실패: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
