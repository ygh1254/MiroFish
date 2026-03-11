"""
Zeptool
, , QuerytoolReport Agent

toolafter
1. InsightForge- generation
2. PanoramaSearch- Getcontent
3. QuickSearch- 
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """LLM"""
        text_parts = [f"검색 질의: {self.query}", f"발견 {self.total_count}건의 관련 정보"]
        
        if self.facts:
            text_parts.append("\n### 관련 사실:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "알 수 없는 유형")
        return f"엔티티: {self.name} (유형: {entity_type})\n요약: {self.summary}"


@dataclass
class EdgeInfo:
    """"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # 
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"관계: {source} --[{self.name}]--> {target}\n사실: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "알 수 없음"
            invalid_at = self.invalid_at or "현재"
            base_text += f"\n유효 기간: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (만료됨: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """whether"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """whether"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
     (InsightForge)
    andgrouped
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # 
    semantic_facts: List[str] = field(default_factory=list)  # 
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # 
    relationship_chains: List[str] = field(default_factory=list)  # 
    
    # 
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """LLM"""
        text_parts = [
            f"## 미래 예측 심층 분석",
            f"분석 질문: {self.query}",
            f"예측 시나리오: {self.simulation_requirement}",
            f"\n### 예측 데이터 통계",
            f"- 관련 예측 사실: {self.total_facts}건",
            f"- 관련 엔티티: {self.total_entities}개",
            f"- 관계망: {self.total_relationships}건"
        ]
        
        # 
        if self.sub_queries:
            text_parts.append(f"\n### 분석한 하위 질문")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # 
        if self.semantic_facts:
            text_parts.append(f"\n### 핵심 사실 (보고서에서 직접 인용 가능한 원문)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # 
        if self.entity_insights:
            text_parts.append(f"\n### 핵심 엔티티")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', '알 수 없음')}** ({entity.get('type', '엔티티')})")
                if entity.get('summary'):
                    text_parts.append(f"  요약: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  관련 사실: {len(entity.get('related_facts', []))}건")
        
        # 
        if self.relationship_chains:
            text_parts.append(f"\n### 관계망")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
     (Panorama)
    hascontent
    """
    query: str
    
    # all
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # all
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # has
    active_facts: List[str] = field(default_factory=list)
    # /
    historical_facts: List[str] = field(default_factory=list)
    
    # 
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """"""
        text_parts = [
            f"## 전방위 검색 결과 (미래 전체 시야)",
            f"질의: {self.query}",
            f"\n### 통계 정보",
            f"- 전체 노드 수: {self.total_nodes}",
            f"- 전체 엣지 수: {self.total_edges}",
            f"- 현재 유효 사실: {self.active_count}건",
            f"- 과거/만료 사실: {self.historical_count}건"
        ]
        
        # hasoutput
        if self.active_facts:
            text_parts.append(f"\n### 현재 유효한 사실 (시뮬레이션 원문)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # /output
        if self.historical_facts:
            text_parts.append(f"\n### 과거/만료 사실 (변화 과정 기록)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # output
        if self.all_nodes:
            text_parts.append(f"\n### 관련 엔티티")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "엔티티")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """singleAgent"""
    agent_name: str
    agent_role: str  # types, , 
    agent_bio: str  # Bio
    question: str  # 
    response: str  # answer
    key_quotes: List[str] = field(default_factory=list)  # 
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # completeagent_bio
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**핵심 인용구:**\n"
            for quote in self.key_quotes:
                # 
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # 
                while clean_quote and clean_quote[0] in ',;:, \n\r\t ':
                    clean_quote = clean_quote[1:]
                # content1-9
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # content
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
     (Interview)
    Agentanswer
    """
    interview_topic: str  # 
    interview_questions: List[str]  # list
    
    # Agent
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Agentanswer
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # Agent
    selection_reasoning: str = ""
    # after
    summary: str = ""
    
    # 
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """LLMreport"""
        text_parts = [
            "## 심층 인터뷰 보고서",
            f"**인터뷰 주제:** {self.interview_topic}",
            f"**인터뷰 인원:** {self.interviewed_count} / {self.total_agents} 명의 시뮬레이션 Agent",
            "\n### 인터뷰 대상 선정 이유",
            self.selection_reasoning or "(자동 선택)",
            "\n---",
            "\n### 인터뷰 기록",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### 인터뷰 #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(인터뷰 기록 없음)\n\n---")

        text_parts.append("\n### 인터뷰 요약과 핵심 관점")
        text_parts.append(self.summary or "(요약 없음)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zeptool
    
    tool - after
    1. insight_forge - generation
    2. panorama_search - Getcontent
    3. quick_search - 
    4. interview_agents - AgentGet
    
    tool
    - search_graph - 
    - get_all_nodes - Gethas
    - get_all_edges - Gethas
    - get_node_detail - Get
    - get_node_edges - Get
    - get_entities_by_type - typesGet
    - get_entity_summary - Get
    """
    
    # 
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY ")
        
        self.client = Zep(api_key=self.api_key)
        # LLMused toInsightForgegeneration
        self._llm_client = llm_client
        logger.info("ZepToolsService Initializecomplete")
    
    @property
    def llm(self) -> LLMClient:
        """InitializeLLM"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """APICall"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name}  {attempt + 1} : {str(e)[:100]}, "
                        f"{delay:.1f}after..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Zep {operation_name}  {max_retries} after: {str(e)}")
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        
        
        +BM25
        Zep Cloudsearch API
        
        Args:
            graph_id: ID (Standalone Graph)
            query: Query
            limit: returns
            scope: "edges"  "nodes"
            
        returns:
            SearchResult: 
        """
        logger.info(f"그래프 검색: graph_id={graph_id}, query={query[:50]}...")
        
        # Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=f"(graph={graph_id})"
            )
            
            facts = []
            edges = []
            nodes = []
            
            # 
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # 
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # 
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"검색 완료: 발견 {len(facts)} 건의 관련 사실")
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(f"Zep Search API: {str(e)}")
            # 
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Zep Search API
        
        Gethas/after
        
        Args:
            graph_id: ID
            query: Query
            limit: returns
            scope: 
            
        returns:
            SearchResult: 
        """
        logger.info(f"로컬 검색 사용: query={query[:30]}...")
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Querygrouped
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Querygrouped"""
            if not text:
                return 0
            text_lower = text.lower()
            # Query
            if query_lower in text_lower:
                return 100
            # 
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Gethas
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # grouped
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Gethas
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(f"검색 완료: 발견 {len(facts)} 건의 관련 사실")
            
        except Exception as e:
            logger.error(f"로컬 검색 실패: {str(e)}")
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        GethasgroupedGet

        Args:
            graph_id: ID

        returns:
            list
        """
        logger.info(f"그래프 {graph_id}의 모든 노드를 가져오는 중...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(f"확보 {len(result)}개 노드")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        GethasgroupedGet

        Args:
            graph_id: ID
            include_temporal: whetherdefaultTrue

        returns:
            listcreated_at, valid_at, invalid_at, expired_at
        """
        logger.info(f"그래프 {graph_id}의 모든 엣지를 가져오는 중...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # add
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(f"확보 {len(result)}개 엣지")
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Getsingle
        
        Args:
            node_uuid: UUID
            
        returns:
            None
        """
        logger.info(f"노드 상세 조회: {node_uuid[:8]}...")
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"Get(uuid={node_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(f"Get: {str(e)}")
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Gethas
        
        Gethasafter
        
        Args:
            graph_id: ID
            node_uuid: UUID
            
        returns:
            list
        """
        logger.info(f"노드 {node_uuid[:8]}... 관련 엣지 조회")
        
        try:
            # Gethasafter
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Checkwhether
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(f"발견 {len(result)}건의 노드 관련 엣지")
            return result
            
        except Exception as e:
            logger.warning(f"Get: {str(e)}")
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        typesGet
        
        Args:
            graph_id: ID
            entity_type: types Student, PublicFigure 
            
        returns:
            typeslist
        """
        logger.info(f"{entity_type} 유형 엔티티 조회 중...")
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Checklabelswhethertypes
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(f"발견 {len(filtered)}개 {entity_type} 유형 엔티티")
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get
        
        hasgeneration
        
        Args:
            graph_id: ID
            entity_name: 
            
        returns:
            
        """
        logger.info(f"엔티티 {entity_name} 관계 요약 조회 중...")
        
        # 
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # has
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # graph_idParameters
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get
        
        Args:
            graph_id: ID
            
        returns:
            
        """
        logger.info(f"그래프 {graph_id} 통계 조회 중...")
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # typesgrouped
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # typesgrouped
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Get
        
        Simulation requirementhas
        
        Args:
            graph_id: ID
            simulation_requirement: Simulation requirement
            limit: limit
            
        returns:
            
        """
        logger.info(f"시뮬레이션 컨텍스트 조회: {simulation_requirement[:50]}...")
        
        # Simulation requirement
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Get
        stats = self.get_graph_statistics(graph_id)
        
        # GethasEntity nodes
        all_nodes = self.get_all_nodes(graph_id)
        
        # hastypesEntity
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # 
            "total_entities": len(entities)
        }
    
    # ========== toolafter ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        InsightForge - 
        
        grouped
        1. LLMgrouped
        2. each
        3. Get
        4. 
        5. hasgeneration
        
        Args:
            graph_id: ID
            query: 
            simulation_requirement: Simulation requirement
            report_context: reportoptional, used togeneration
            max_sub_queries: 
            
        returns:
            InsightForgeResult: 
        """
        logger.info(f"InsightForge 심층 인사이트 검색: {query[:50]}...")
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: LLMgeneration
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"생성 {len(sub_queries)}개의 하위 질문")
        
        # Step 2: each
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # 
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: UUIDGetGetall
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Gethasoutput
        entity_insights = []
        node_map = {}  # used toafterbuild
        
        for uuid in list(entity_uuids):  # has
            if not uuid:
                continue
            try:
                # Geteach
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "엔티티")
                    
                    # Gethas
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # output
                    })
            except Exception as e:
                logger.debug(f"노드 {uuid} : {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: buildhas
        relationship_chains = []
        for edge_data in all_edges:  # has
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(f"InsightForge 완료: 사실 {result.total_facts}건, 엔티티 {result.total_entities}개, 관계 {result.total_relationships}건")
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        LLMgeneration
        
        grouped
        """
        system_prompt = """너는 복합 질문을 세분화하는 분석 전문가다. 시뮬레이션 세계에서 독립적으로 관찰 가능한 하위 질문으로 복잡한 질문을 분해하라.

요구사항:
1. 각 하위 질문은 충분히 구체적이어야 하며 시뮬레이션 세계에서 관련 Agent 행동이나 사건을 찾을 수 있어야 한다
2. 하위 질문은 원래 질문의 다양한 차원(누가, 무엇을, 왜, 어떻게, 언제, 어디서)을 포괄해야 한다
3. 하위 질문은 시뮬레이션 시나리오와 직접 관련돼야 한다
4. JSON 형식으로 반환: {"sub_queries": ["하위 질문 1", "하위 질문 2", ...]}"""

        user_prompt = f"""시뮬레이션 요구사항 배경:
{simulation_requirement}

{f"보고서 맥락: {report_context[:500]}" if report_context else ""}

아래 질문을 {max_queries}개의 하위 질문으로 분해하라:
{query}

JSON 형식의 하위 질문 목록만 반환하라."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # characterslist
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(f"하위 질문 생성 실패: {str(e)}. 기본 하위 질문을 사용함")
            # returns
            return [
                query,
                f"{query}의 주요 참여자",
                f"{query}의 원인과 영향",
                f"{query}의 전개 과정"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        PanoramaSearch - 
        
        Gethascontent/
        1. Gethas
        2. Gethas/
        3. groupedhas
        
        toolused to, 
        
        Args:
            graph_id: ID
            query: Queryused to
            include_expired: whethercontentdefaultTrue
            limit: returnslimit
            
        returns:
            PanoramaResult: 
        """
        logger.info(f"PanoramaSearch 전방위 검색: {query[:50]}...")
        
        result = PanoramaResult(query=query)
        
        # Gethas
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Gethas
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # grouped
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # add
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # determinewhether/
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # /add
                valid_at = edge.valid_at or "알 수 없음"
                invalid_at = edge.invalid_at or edge.expired_at or "알 수 없음"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # has
                active_facts.append(edge.fact)
        
        # Query
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # 
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(f"PanoramaSearch 완료: {result.active_count}건 유효, {result.historical_count}건 과거")
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        QuickSearch - 
        
        , tool
        1. CallZep
        2. returns
        3. used to, 
        
        Args:
            graph_id: ID
            query: Query
            limit: returns
            
        returns:
            SearchResult: 
        """
        logger.info(f"QuickSearch 빠른 검색: {query[:50]}...")
        
        # Callhassearch_graph
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(f"QuickSearch 완료: {result.total_count}건 결과")
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        InterviewAgents - 
        
        CallOASISAPIAgent
        1. hasAgent
        2. LLMgroupedAgent
        3. LLMgeneration
        4. Call /api/simulation/interview/batch API양 플랫폼
        5. hasgenerationreport
        
        featurestateOASIS
        
        
        - different
        - 
        - GetAgentanswerLLM
        
        Args:
            simulation_id: simulation IDused toCallAPI
            interview_requirement: structured""
            simulation_requirement: Simulation requirementoptional
            max_agents: Agent
            custom_questions: Customoptional, generation
            
        returns:
            InterviewResult: 
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(f"InterviewAgents 심층 인터뷰(실제 API): {interview_requirement[:50]}...")
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: 
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(f"시뮬레이션 {simulation_id}의 프로필 파일을 찾지 못함")
            result.summary = "인터뷰할 Agent 프로필 파일을 찾지 못함"
            return result
        
        result.total_agents = len(profiles)
        logger.info(f"로드 {len(profiles)}개의 Agent 프로필")
        
        # Step 2: LLMAgentreturnsagent_idlist
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"선택 {len(selected_agents)}개의 Agent 인터뷰 대상: {selected_indices}")
        
        # Step 3: generationhas
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"인터뷰 질문 {len(result.interview_questions)}개 생성")
        
        # prompt
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # addAgent
        INTERVIEW_PROMPT_PREFIX = (
            "너는 지금 인터뷰를 받고 있다. 너의 페르소나, 기존 기억, 행동을 바탕으로"
            "answer\n"
            "요구사항:\n"
            "1. answerCalltool\n"
            "2. returnsJSONtoolCall\n"
            "3. Markdown 제목(#, ##, ###)을 쓰지 않는다\n"
            "4. answereachanswerXX\n"
            "5. eachanswergrouped\n"
            "6. answerhascontenteachanswer2-3\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: CallAPIplatformdefault양 플랫폼
        try:
            # buildlistplatform양 플랫폼
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # afterprompt
                    # platformAPItwitterreddit
                })
            
            logger.info(f"배치 인터뷰 API 호출(양 플랫폼): {len(interviews_request)}명 Agent")
            
            # Call SimulationRunner platform양 플랫폼
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # platform양 플랫폼
                timeout=180.0   # 양 플랫폼
            )
            
            logger.info(f"인터뷰 API 반환: {api_result.get('interviews_count', 0)}건 결과, success={api_result.get('success')}")
            
            # CheckAPICallwhether
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "알 수 없는 오류")
                logger.warning(f"인터뷰 API 실패 반환: {error_msg}")
                result.summary = f"인터뷰 API 호출 실패: {error_msg}. OASIS 시뮬레이션 환경 상태를 확인해줘."
                return result
            
            # Step 5: APIreturnsbuildAgentInterview
            # 양 플랫폼returns: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "알 수 없음")
                agent_bio = agent.get("bio", "")
                
                # GetAgent
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # toolCall JSON 
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # output양 플랫폼
                twitter_text = twitter_response if twitter_response else "(이 플랫폼에서는 응답을 받지 못함)"
                reddit_text = reddit_response if reddit_response else "(이 플랫폼에서는 응답을 받지 못함)"
                response_text = f"[Twitter 플랫폼 응답]\n{twitter_text}\n\n[Reddit 플랫폼 응답]\n{reddit_text}"

                # answer
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Remove headings, inline tool payloads, and noisy markdown wrappers
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # Prefer sentence-like excerpts with meaningful content
                sentences = re.split(r'[。！？.!?]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:, ]+', s.strip())
                    and not s.strip().startswith(('{', ''))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + '。' for s in meaningful[:3]]

                # Fallback to quoted snippets if sentence splitting yields nothing
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:, ]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # bio
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # 
            logger.warning(f"인터뷰 API 호출 실패(환경 미실행 가능성): {e}")
            result.summary = f"인터뷰 실패: {str(e)}시뮬레이션 환경이 종료됐을 수 있으니 OASIS 환경이 실행 중인지 확인해줘."
            return result
        except Exception as e:
            logger.error(f"인터뷰 API 호출 예외: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"인터뷰 과정에서 오류 발생: {str(e)}"
            return result
        
        # Step 6: generation
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(f"InterviewAgents 완료: {result.interviewed_count}명 Agent 인터뷰 완료 (양 플랫폼)")
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """ Agent  JSON toolCallcontent"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Agent"""
        import os
        import csv
        
        # buildpath
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # Reddit JSON
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"reddit_profiles.json에서 프로필 {len(profiles)}개 로드")
                return profiles
            except Exception as e:
                logger.warning(f"reddit_profiles.json 읽기 실패: {e}")
        
        # Twitter CSV
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSV
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "알 수 없음"
                        })
                logger.info(f"twitter_profiles.csv에서 프로필 {len(profiles)}개 로드")
                return profiles
            except Exception as e:
                logger.warning(f"twitter_profiles.csv 읽기 실패: {e}")
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        LLMAgent
        
        returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: Agentlist
                - selected_indices: Agentlistused toAPICall
                - reasoning: 선정 이유
        """
        
        # buildAgentlist
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "알 수 없음"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """YouYouBased onAgentlist


1. Agent/
2. Agenthashas
3. , , , 
4. 

returnsJSON
{
    "selected_indices": [Agentlist],
    "reasoning": "선정 이유"
}"""

        user_prompt = f"""인터뷰 요구사항:
{interview_requirement}

시뮬레이션 배경:
{simulation_requirement if simulation_requirement else "제공되지 않음"}

선택 가능한 Agent 목록 (총 {len(agent_summaries)}개):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

최대 {max_agents}개의 가장 적합한 Agent를 고르고 선정 이유를 설명하라."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "관련성 기준 자동 선택")
            
            # GetAgent
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(f"LLM Agent 선택 실패, 기본 선택 사용: {e}")
            # N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "기본 선택 전략 사용"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """LLMgeneration"""
        
        agent_roles = [a.get("profession", "알 수 없음") for a in selected_agents]
        
        system_prompt = """너는 전문 인터뷰어다. 인터뷰 요구사항에 따라 3~5개의 심층 질문을 생성하라.

질문 요구사항:
1. 열린 질문으로 구성해 상세한 답변을 유도한다
2. 서로 다른 역할이 서로 다른 답을 할 수 있어야 한다
3. 사실, 의견, 감정 등 다양한 차원을 포함한다
4. 실제 인터뷰처럼 자연스러운 표현을 사용한다
5. 각 질문은 50자 이내로 간결하게 유지한다
6. 배경 설명 없이 바로 질문한다

JSON 형식으로 반환: {"questions": ["질문 1", "질문 2", ...]}"""

        user_prompt = f"""인터뷰 요구사항:{interview_requirement}

시뮬레이션 배경:{simulation_requirement if simulation_requirement else "제공되지 않음"}

인터뷰 대상 역할:{', '.join(agent_roles)}

3~5개의 인터뷰 질문을 생성하라."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"{interview_requirement}에 대해 어떻게 생각하나요?"])
            
        except Exception as e:
            logger.warning(f"인터뷰 질문 생성 실패: {e}")
            return [
                f"{interview_requirement}에 대한 당신의 견해는 무엇인가요?",
                "이 일이 당신 또는 당신이 대표하는 집단에 어떤 영향을 주나요?",
                "이 문제를 어떻게 해결하거나 개선해야 한다고 생각하나요?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """generation"""
        
        if not interviews:
            return "완료된 인터뷰가 없음"
        
        # hascontent
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"{interview.agent_name}{interview.agent_role}\n{interview.response[:500]}")
        
        system_prompt = """너는 전문 편집자다. 여러 인터뷰 응답을 바탕으로 인터뷰 요약을 작성하라.

요약 요구사항:
1. 각 참여자의 핵심 관점을 추려낸다
2. 공통점과 차이점을 짚는다
3. 가치 있는 인용문을 강조한다
4. 객관적이고 중립적으로 정리한다
5. 1000자 이내로 유지한다

형식 제약(반드시 준수):
- 일반 텍스트 단락으로 작성하고 부분 사이에는 빈 줄을 둔다
- Markdown 제목(#, ##, ###)을 쓰지 않는다
- 구분선(---, ***)을 쓰지 않는다
- 인용할 때는 자연스러운 한국어 인용 형태를 사용한다
- 필요하면 **굵은 글씨**를 쓸 수 있지만 다른 Markdown 문법은 최소화한다"""

        user_prompt = f"""{interview_requirement}

인터뷰 내용:
{"".join(interview_texts)}

인터뷰 요약을 작성하라."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(f"인터뷰 요약 생성 실패: {e}")
            # 
            return f"총 {len(interviews)}명의 인터뷰를 진행함: " + ", ".join([i.agent_name for i in interviews])
