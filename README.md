# BrickGraphAgent: OKF 기반 지식 그래프와 ActiveGraph를 결합한 자가 개선 에이전트 시스템

> *"Architecture should be derived, not designed. The graph is a control surface, not just a lookup layer."*

---

## 📌 개요

**BrickGraphAgent**는 OKF(Open Knowledge Format) 지식 베이스와 ActiveGraph 이벤트-소싱 런타임을 통합한 자가 개선 에이전트 시스템입니다. 이 시스템은 보고서/제안서의 논리적 모순을 검출하고, 계층적 지식 체계를 **brick-by-brick**으로 구축하며, 실시간 적응(runtime adaptation)을 통해 지속적으로 개선됩니다.

### 핵심 원칙

1. **아키텍처는 설계되지 않고 도출되어야 한다** — 에이전트의 행동 궤적(traces)에서 시스템이 진화합니다.
2. **결정론적 영역과 에이전틱 영역을 분리** — 수학/통계는 결정론적 파이프라인으로, 추론과 발견은 에이전트가 담당합니다.
3. **하나의 에이전트가 추론의 전 과정을 종단적으로 소유** — 컨텍스트 손실과 일관성 없는 추론을 방지합니다.
4. **그래프는 제어 표면(control surface)이지 조회 계층이 아니다** — 지식 그래프가 에이전트의 행동을 통제(govern)합니다.

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE (OKF KB Graph)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐│
│  │  온톨로지    │  │  정책 규칙   │  │  증거/근거 그래프               ││
│  │ (Concept,   │  │ (추론 패턴,  │  │ (Evidence-CLAIM 연결)           ││
│  │  Relation)  │  │  검증 규칙)  │  │                                 ││
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ACTIVE GRAPH RUNTIME (Event-Sourced)                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     이벤트 버스 (Event Bus)                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │  │
│  │  │ Brick 01   │  │ Brick 02   │  │ Brick 03   │  │ Brick 04  │ │  │
│  │  │ (Injest)   │──│ (Compile)  │──│ (Ask)      │──│ (Browse)  │ │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              이벤트 로그 (storage/events.log)                    │  │
│  │  - 모든 에이전트 행동의 타입화된 추적 (immutable append-only)   │  │
│  │  - 모델/도구 응답 캐시로 정확한 재생(replay) 보장               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      에이전트 스위트 (Agent Suite)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Builder    │  │   Critic     │  │  Improver    │  │  Curator   │  │
│  │ (지식 조립)  │  │ (모순 탐지)  │  │ (구조 최적화)│  │ (인간 검토)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 디렉토리 구조

```
brick-graph-agent/
├── .okf/                              # 📂 전체 시스템의 지식베이스 (Knowledge Base)
│   ├── index.md                       # KB 메인 인덱스 및 아키텍처 맵
│   ├── log.md                         # ISO 날짜 기반 변경 이력 (최신순)
│   │
│   ├── 01_brick_agent/                # 🧱 에이전트 코어 설계서
│   │   ├── system_overview.md         # ActiveGraph 버스 및 토픽 구조
│   │   ├── brick_injest.md            # Brick 01 (Injest) 기능 명세
│   │   ├── brick_compile.md           # Brick 02 (Compile) 명세
│   │   ├── brick_ask.md               # Brick 03 (Ask) 명세
│   │   └── brick_browse.md            # Brick 04 (Browse) 명세
│   │
│   ├── 02_agent_roles/                # 🧠 에이전트 역할별 설계서
│   │   ├── builder.md                 # Builder 에이전트 (지식 조립)
│   │   ├── critic.md                  # Critic 에이전트 (모순 탐지)
│   │   ├── improver.md                # Improver 에이전트 (구조 최적화)
│   │   └── curator.md                 # Curator (인간 개발자 역할)
│   │
│   ├── 03_domain_logic/               # 📊 도메인 논리 및 추론 규칙
│   │   ├── contradiction_patterns.md  # 논리적 모순 패턴 정의
│   │   ├── inference_rules.md         # 추론 규칙 (ENTAILS, SUPPORTS 등)
│   │   └── trust_tiers.md             # 신뢰 등급 체계
│   │
│   └── 04_pdf_rag/                    # 📄 PDF RAG 도메인 지식
│       ├── pipeline_flow.md           # PDF 파싱부터 마크다운 변환 흐름
│       ├── parser_marker.md           # Marker 라이브러리 연동 가이드
│       └── chunking_strategy.md       # 청킹 전략 문서
│
├── src/                               # 📂 실행 가능한 파이썬 소스코드
│   ├── main.py                        # 에이전트 전체 실행 및 이벤트 루프 진입점
│   │
│   ├── active_graph/                  # 🤖 ActiveGraph 1.10.0 이벤트 버스 레이어
│   │   ├── __init__.py
│   │   ├── event_store.py             # 불변 이벤트 로그 기록 및 구독 관리
│   │   ├── topics.py                  # 이벤트 토픽(Enum) 및 스키마 정의
│   │   └── cache.py                   # 모델/도구 응답 캐시 (재생용)
│   │
│   ├── agent_core/                    # 🧱 에이전트 코어 벽돌 (Bricks)
│   │   ├── injestor.py                # Brick 01: OKF 변경 감지 및 이벤트 발행
│   │   ├── compiler.py                # Brick 02: 지식 그래프 컴파일
│   │   ├── questioner.py              # Brick 03: 컨텍스트 그래프 기반 추론
│   │   └── browser.py                 # Brick 04: 실시간 그래프 시각화
│   │
│   ├── agents/                        # 🧠 전문 에이전트 구현체
│   │   ├── base_agent.py              # 에이전트 추상 베이스 클래스
│   │   ├── builder.py                 # Builder: 원자적 개념 → 복합 시스템 조립
│   │   ├── critic.py                  # Critic: 논리적 모순 탐지
│   │   ├── improver.py                # Improver: 트리 구조 최적화 제안
│   │   └── oracle.py                  # Oracle: 실패 레짐 분류기 (진단용)
│   │
│   ├── pdf_rag/                       # 📄 PDF 처리 도메인 벽돌
│   │   ├── pdf_processor.py           # Marker/MinerU 연동 PDF 마크다운 분해기
│   │   ├── vector_store.py            # 임베딩 및 벡터 저장소
│   │   └── okf_extractor.py           # PDF → OKF 마크다운 추출 스크립트
│   │
│   └── tools/                         # 🔧 유틸리티 도구
│       ├── okf_validator.py           # OKF KB 검사기 (okf-validate 래퍼)
│       ├── okf_visualizer.py          # OKF KB 시각화 도구
│       └── okf_migrate.py             # OKF v0.1 → v0.2 마이그레이션
│
├── storage/                           # 📂 런타임 데이터 및 파일 저장소
│   ├── events.log                     # ActiveGraph 이벤트 로그 (append-only)
│   ├── events.log.index               # 이벤트 인덱스 (빠른 조회용)
│   ├── source_pdfs/                   # 원본 PDF 파일 보관소
│   └── extracted_md/                  # PDF에서 추출된 마크다운/이미지
│
├── dist/                              # 📂 브라우징 툴 출력 폴더
│   └── index.html                     # 실시간 지식 그래프 대시보드
│
├── experiments/                       # 📊 논문 실험용 스크립트 및 데이터
│   ├── splits/                        # 데이터셋 분할 (OPTIMIZE/CONFIRM)
│   ├── results/                       # 실험 결과 및 로그
│   └── analysis/                      # 통계 분석 노트북
│
├── tests/                             # 🧪 단위 및 통합 테스트
│   ├── test_bricks.py                 # 벽돌 단위 테스트
│   ├── test_agents.py                 # 에이전트 동작 테스트
│   └── fixtures/                      # 테스트 픽스처 (OKF 번들)
│
├── requirements.txt                   # 의존성 라이브러리
├── pyproject.toml                     # 프로젝트 메타데이터 및 빌드 설정
├── README.md                          # 본 문서
└── CLAUDE.md                          # Claude Code 플러그인용 가이드
```

---

## 🧱 Brick-by-Brick 에이전트 구조

### Brick 01: Event-driven Knowledge Injestor (지식 수집)

```python
# src/agent_core/injestor.py
@llm_behavior(pattern="""
    MATCH (f:File)
    WHERE f.path CONTAINS '.okf/' AND f.modified_at > last_sync
    RETURN f
""")
def injestor_agent(graph, matched_files):
    """
    OKF KB의 변경사항을 감지하고 KNOWLEDGE_MODIFIED 이벤트를 발행.
    새 파일이 추가되거나 기존 파일이 수정되면 이벤트를 생성.
    """
    for file in matched_files:
        event = {
            'type': 'KNOWLEDGE_MODIFIED',
            'payload': {
                'path': file.path,
                'frontmatter': parse_yaml(file.content),
                'body': file.content,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        graph.emit(event)
        graph.add_node(file.path, type='KnowledgeNode', modified_at=file.modified_at)
```

### Brick 02: Dynamic Graph Compiler (지식 결합)

```python
# src/agent_core/compiler.py
@llm_behavior(pattern="""
    MATCH (e:Event)
    WHERE e.type = 'KNOWLEDGE_MODIFIED' AND e.processed = false
    RETURN e
""")
def compiler_agent(graph, events):
    """
    KNOWLEDGE_MODIFIED 이벤트를 수신하여 지식 그래프를 실시간 컴파일.
    파일 간 references 관계를 탐지하고, 온톨로지 기반 링크 생성.
    """
    for event in events:
        path = event.payload['path']
        references = extract_references(event.payload['body'])
        
        for ref in references:
            graph.add_edge(path, ref, type='REFERENCES')
        
        # OKF v0.2 타입 추론
        node_type = event.payload['frontmatter'].get('type')
        if node_type:
            graph.add_metadata(path, 'type', node_type)
        
        graph.mark_processed(event.id)
        graph.emit({'type': 'GRAPH_COMPILED', 'node': path})
```

### Brick 03: Context-Aware Ask Engine (질의응답 및 추론)

```python
# src/agent_core/questioner.py
@llm_behavior(pattern="""
    MATCH (q:Query)-[:ASKED]->()
    RETURN q
""")
def questioner_agent(graph, queries):
    """
    사용자 질문을 받아 관련 OKF 문서와 컴파일된 그래프 컨텍스트를 함께
    프롬프트에 주입하여 답변 생성.
    """
    for query in queries:
        # 1. 관련 노드 검색 (그래프 기반)
        related_nodes = graph.query("""
            MATCH (n:KnowledgeNode)
            WHERE n.text CONTAINS $keywords
            RETURN n, 
                   [(n)-[:REFERENCES|SUPPORTS|CONTRADICTS]-(m) | m] as neighbors
        """, keywords=query.keywords)
        
        # 2. 컨텍스트 구성
        context = build_context(related_nodes)
        
        # 3. LLM 호출 (캐시 가능)
        answer = llm.generate(prompt=query.prompt, context=context)
        
        # 4. 응답 이벤트 발행 (provenance 포함)
        graph.emit({
            'type': 'RESPONSE_GENERATED',
            'payload': {
                'query': query.text,
                'answer': answer,
                'sources': [n.path for n in related_nodes]
            }
        })
```

### Brick 04: Real-time Live Browser (실시간 시각화)

```python
# src/agent_core/browser.py
@llm_behavior(pattern="""
    MATCH (e:Event)
    WHERE e.type IN ['KNOWLEDGE_MODIFIED', 'GRAPH_COMPILED', 'RESPONSE_GENERATED']
    AND e.timestamp > last_broadcast
    RETURN e
""")
def browser_agent(graph, events):
    """
    이벤트 로그를 웹소켓으로 브로드캐스트하여 실시간 그래프 업데이트.
    노드 및 엣지 변경사항을 시각적 대시보드에 반영.
    """
    for event in events:
        ws_broadcast({
            'event_type': event.type,
            'node': event.payload.get('node'),
            'edge': event.payload.get('edge'),
            'timestamp': event.timestamp
        })
```

---

## 🧠 전문 에이전트 스위트

### Builder Agent (지식 조립)

```python
# src/agents/builder.py
class BuilderAgent(BaseAgent):
    """
    OKF KB에서 원자적 개념(AtomicConcept)을 감지하고,
    선행 조건(PREREQUISITE_OF)이 충족되면 복합 개념(CompositeConcept)으로 조립.
    """
    
    @llm_behavior(pattern="""
        MATCH (c:Concept)
        WHERE c.type = 'AtomicConcept' 
          AND NOT exists((c)-[:COMPOSED_OF]->())
          AND all(p IN (c)-[:PREREQUISITE_OF]-(pr) WHERE pr.learned = true)
        RETURN c
    """)
    def assemble_composite(self, graph, atomic_concepts):
        for concept in atomic_concepts:
            prerequisites = self.get_prerequisites(concept)
            composite = self.create_composite(concept, prerequisites)
            
            if composite:
                graph.add_node(composite, type='CompositeConcept')
                graph.add_edge(composite, concept, 'COMPOSED_OF')
                for p in prerequisites:
                    graph.add_edge(composite, p, 'COMPOSED_OF')
                
                graph.emit({
                    'type': 'COMPOSITE_ASSEMBLED',
                    'payload': {'composite': composite.id, 'components': len(prerequisites)+1}
                })
```

### Critic Agent (모순 탐지)

```python
# src/agents/critic.py
class CriticAgent(BaseAgent):
    """
    그래프에서 논리적 모순 패턴을 탐지하고 진단 보고서를 생성.
    - 직접 모순: CONTRADICTS 관계
    - 근거 없는 주장: BASED_ON 관계 누락
    - 순환 논증: 순환 의존성 탐지
    - 전제-결론 불일치: ENTAILS 규칙 위반
    """
    
    @llm_behavior(pattern="""
        MATCH (a:Claim)-[:CONTRADICTS]->(b:Claim)
        RETURN a, b
    """)
    def detect_direct_contradiction(self, graph, contradictions):
        for a, b in contradictions:
            report = {
                'type': 'DIRECT_CONTRADICTION',
                'claim_a': a.id,
                'claim_b': b.id,
                'severity': 'high',
                'suggestion': f"Resolve contradiction between '{a.title}' and '{b.title}'"
            }
            graph.emit({'type': 'CONTRADICTION_FOUND', 'payload': report})
    
    @llm_behavior(pattern="""
        MATCH (c:Claim)
        WHERE NOT exists((c)-[:BASED_ON]->())
        RETURN c
    """)
    def detect_unsupported_claim(self, graph, claims):
        for claim in claims:
            report = {
                'type': 'UNSUPPORTED_CLAIM',
                'claim': claim.id,
                'severity': 'medium',
                'suggestion': f"Add evidence for '{claim.title}'"
            }
            graph.emit({'type': 'CONTRADICTION_FOUND', 'payload': report})
```

### Improver Agent (구조 최적화)

```python
# src/agents/improver.py
class ImproverAgent(BaseAgent):
    """
    기존 지식 트리 구조를 분석하고 개선안을 제안.
    - 재구성(Recomposition): 더 효율적인 계층으로 재배치
    - 분할(Splitting): 과도하게 복잡한 개념을 분해
    - 병합(Merging): 지나친 세분화를 통합
    - 선행 조건 재정렬: 더 논리적인 학습 순서 제안
    """
    
    @llm_behavior(pattern="""
        MATCH (s:System)-[:COMPOSED_OF*]->(c:Concept)
        RETURN s, collect(c) as components
    """)
    def analyze_and_improve(self, graph, systems):
        for system in systems:
            components = system['components']
            depth_analysis = self.analyze_tree_depth(components)
            
            improvement = self.generate_improvement(
                current_structure=components,
                depth_analysis=depth_analysis
            )
            
            if improvement:
                graph.add_node(improvement, type='ImprovementSuggestion')
                graph.add_edge(system, improvement, 'SUGGESTS_IMPROVEMENT')
                graph.emit({
                    'type': 'IMPROVEMENT_PROPOSED',
                    'payload': improvement.to_dict()
                })
    
    def generate_improvement(self, current_structure, depth_analysis):
        # LLM 호출하여 개선안 생성 (held-out gate 적용)
        prompt = f"""
        Analyze this knowledge tree structure and propose improvements:
        Current structure: {current_structure}
        Depth analysis: {depth_analysis}
        
        Suggest one of: recomposition, splitting, merging, or reordering.
        """
        return llm.generate(prompt)
```

### Oracle Agent (실패 레짐 분류기)

```python
# src/agents/oracle.py
class OracleAgent(BaseAgent):
    """
    실패한 평가를 진단하고 개선할 수 있는 레짐(regime)으로 분류.
    - assemble-internal: 증거가 컨텍스트에 있지만 잘못 해석
    - budget-truncation: 증거가 컨텍스트 예산에서 탈락
    - retrieval-signal-gap: 검색 점수가 충분하지 않음
    - scoring-error: 평가 자체 오류
    """
    
    def classify_failure(self, question, evidence, answer):
        if self.evidence_present_in_context(evidence) and not self.is_correct(answer):
            return 'assemble-internal'  # → reader-prompt-transform 적용
        elif self.evidence_dropped_at_budget(evidence):
            return 'budget-truncation'  # → score-transform 또는 assembly-transform
        elif not self.evidence_retrieved(evidence):
            return 'retrieval-signal-gap'  # → 개선 불가 (true wall)
        else:
            return 'scoring-error'
```

---

## 🔧 필수 도구 스크립트

### PDF → OKF 마크다운 추출기

```python
# src/pdf_rag/okf_extractor.py
"""
PDF 파일을 OKF v0.2 규격의 마크다운 문서로 변환하는 스크립트.

사용법:
    python -m src.pdf_rag.okf_extractor --input storage/source_pdfs/doc.pdf --output .okf/
"""
import argparse
from pathlib import Path
from marker.converters import PdfConverter
from marker.models import create_model_dict

def extract_pdf_to_okf(pdf_path: Path, output_dir: Path):
    """PDF를 OKF 마크다운으로 변환"""
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    
    # OKF v0.2 프론트매터 생성
    frontmatter = f"""---
type: Document
title: {pdf_path.stem}
description: Extracted from PDF
generated:
  by: process:pdf_extractor
  at: {datetime.utcnow().isoformat()}
status: unverified
---
"""
    # 마크다운 본문
    content = f"{frontmatter}\n\n{rendered.markdown}"
    
    # OKF 파일로 저장
    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(content)
    print(f"✅ Extracted: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="PDF 파일 경로")
    parser.add_argument("--output", default=".okf/", help="출력 디렉토리")
    args = parser.parse_args()
    extract_pdf_to_okf(Path(args.input), Path(args.output))
```

### OKF KB 검사기

```python
# src/tools/okf_validator.py
"""
OKF KB의 구조적 무결성과 OKF v0.2 규격 준수 여부를 검사하는 도구.

사용법:
    python -m src.tools.okf_validator .okf/ --strict
"""
import sys
from pathlib import Path
import yaml
import re

def validate_okf_bundle(bundle_path: Path, strict: bool = False):
    """OKF 번들 검증"""
    errors = []
    warnings = []
    
    # index.md 존재 확인
    index_path = bundle_path / "index.md"
    if not index_path.exists():
        errors.append("❌ .okf/index.md not found")
    
    # 모든 .md 파일 검사
    for md_file in bundle_path.rglob("*.md"):
        content = md_file.read_text()
        
        # YAML 프론트매터 파싱
        if not content.startswith("---"):
            errors.append(f"❌ {md_file}: missing YAML frontmatter")
            continue
        
        try:
            _, frontmatter_str, body = content.split("---", 2)
            fm = yaml.safe_load(frontmatter_str)
        except Exception as e:
            errors.append(f"❌ {md_file}: invalid YAML - {e}")
            continue
        
        # OKF v0.2 필수 필드 검사
        if 'type' not in fm:
            errors.append(f"❌ {md_file}: missing 'type' field")
        
        # 권장 필드 경고
        if 'description' not in fm:
            warnings.append(f"⚠️ {md_file}: missing 'description'")
        
        if 'generated' not in fm:
            warnings.append(f"⚠️ {md_file}: missing 'generated' metadata")
    
    # 결과 출력
    print(f"📊 OKF Validation Results: {bundle_path}")
    print(f"   Errors: {len(errors)}, Warnings: {len(warnings)}")
    
    for e in errors:
        print(e)
    for w in warnings:
        print(w)
    
    if strict and errors:
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", help="OKF 번들 디렉토리 경로")
    parser.add_argument("--strict", action="store_true", help="엄격 모드")
    args = parser.parse_args()
    validate_okf_bundle(Path(args.bundle), args.strict)
```

---

## 🔬 논문 실험 워크플로우

### 실험 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                     데이터 준비 (500문서 데이터셋)               │
│  - 보고서/제안서 수집 (연구 제안서, 비즈니스 리포트)           │
│  - 인간 주석: 논리적 오류 레이블링 (모순, 근거 부재 등)        │
│  - OPTIMIZE (50) / CONFIRM (100) / TEST (350) 분할            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              BrickGraphAgent 실행 (5개 시드 분할)               │
│  - Builder: 원자적 개념 추출 및 계층적 조립                    │
│  - Critic: 논리적 모순 탐지 및 진단                            │
│  - Improver: 트리 구조 최적화 제안                             │
│  - Oracle: 실패 레짐 분류 → 적절한 seam으로 라우팅             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                Held-Out Gate (보류 평가 게이트)                 │
│  - 후보 수정안이 CONFIRM 분할에서 성능 향상?                   │
│  - 향상 없음 → 폐기 (discard)                                 │
│  - 향상 있음 → 승격 (promote)                                 │
│  - 모든 레짐 소진 → 종료                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     결과 측정 및 분석                           │
│  - McNemar 검정 (wrong→right vs right→wrong)                  │
│  - Bootstrap 신뢰구간                                         │
│  - 레짐별 국소화(localization) 분석                           │
│  - 오버-프로모션(over-promotion) 탐지                        │
└─────────────────────────────────────────────────────────────────┘
```

### 실험 실행 명령어

```bash
# 전체 실험 실행
python -m experiments.run_experiment \
    --data-dir ./experiments/data \
    --splits 5 \
    --seeds 7 11 23 5 101

# 특정 시드 분석
python -m experiments.analyze_split \
    --split seed_7 \
    --output ./experiments/results/seed_7/

# 통계 테스트
python -m experiments.statistical_tests \
    --results ./experiments/results/ \
    --output ./experiments/analysis/report.md
```

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 3.11+ 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# ActiveGraph 설치 (PyPI)
pip install activegraph==1.10.0
```

### 2. OKF KB 초기화

```bash
# OKF 번들 생성
python -m src.tools.okf_init .okf --title "BrickGraphAgent KB"

# 샘플 문서 추가
python -m src.pdf_rag.okf_extractor --input sample.pdf --output .okf/
```

### 3. 에이전트 실행

```bash
# 메인 에이전트 실행 (이벤트 루프)
python src/main.py --bundle .okf --storage storage/

# 특정 브릭만 테스트
python -m src.agent_core.injestor --watch .okf/
python -m src.agent_core.compiler --listen localhost:8000
```

### 4. 웹 대시보드 확인

```bash
# 그래프 시각화 생성
python -m src.tools.okf_visualizer .okf -o dist/index.html

# 브라우저로 열기
open dist/index.html  # 또는 start dist/index.html
```

### 5. OKF KB 검증

```bash
# 검증 실행
python -m src.tools.okf_validator .okf --strict

# CI 연동 예시 (GitHub Actions)
# .github/workflows/validate.yml에 추가:
# - uses: scaccogatto/okf-skills@v1
#   with:
#     bundle: .okf
#     strict: "true"
```

---

## 📊 논문 실험 재현

### 데이터셋 준비

```bash
# 실험 데이터 다운로드 (예시)
wget https://example.com/brickgraph-benchmark.zip
unzip brickgraph-benchmark.zip -d experiments/data/

# OPTIMIZE/CONFIRM 분할 생성
python -m experiments.prepare_splits \
    --data experiments/data/ \
    --output experiments/splits/ \
    --seeds 7 11 23 5 101
```

### 실험 실행 및 결과 수집

```bash
# 모든 시드에 대해 실험 실행
for seed in 7 11 23 5 101; do
    python -m experiments.run_split \
        --seed $seed \
        --optimize-size 50 \
        --confirm-size 100 \
        --output experiments/results/seed_${seed}/
done

# 결과 집계
python -m experiments.aggregate_results \
    --input experiments/results/ \
    --output experiments/analysis/table1.md
```

### 통계 테스트

```python
# experiments/statistical_tests.py
import pandas as pd
from scipy.stats import mcnemar
from sklearn.utils import resample

def run_mcnemar_test(baseline_results, post_results):
    """McNemar 검정 실행"""
    # 혼동 행렬 생성
    # baseline_correct, post_incorrect 등
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    return result.pvalue

def bootstrap_ci(baseline, post, n_resamples=10000):
    """부트스트랩 신뢰구간 계산"""
    diffs = [post[i] - baseline[i] for i in range(len(baseline))]
    ci_lower, ci_upper = resample(diffs, n_resamples=n_resamples).confidence_interval
    return ci_lower, ci_upper
```

---

## 📚 OKF v0.2 규격 참조

이 프로젝트는 OKF v0.2 규격을 준수합니다. 자세한 사항은 다음을 참조하세요:

- **공식 스펙**: [OKF v0.2 SPEC.md](https://github.com/scaccogatto/okf-skills/blob/main/skills/okf/reference/SPEC.md)
- **예제 번들**: [examples/sample-bundle/](https://github.com/scaccogatto/okf-skills/tree/main/examples/sample-bundle)

### 필수 필드

```yaml
---
type: Concept | Service | Decision | Document | AgentRole | ...
# 모든 OKF 파일은 'type' 필드를 가져야 함 (필수)
```

### 권장 필드

```yaml
---
type: Concept
title: Atomic Concept Name
description: Brief description of the concept
status: draft | stable | deprecated
generated:
  by: agent:builder/1.0
  at: 2026-08-05T10:00:00Z
verified:
  - by: human:curator
    at: 2026-08-06T09:00:00Z
stale_after: 2026-12-31
sources:
  - id: source-doc
    resource: https://example.com/doc
    title: Source Document
---
```

---

## 🤝 기여 가이드

### 개발자 역할: 큐레이터(Curator)

개발자의 역할은 더 이상 모든 로직을 직접 코딩하는 것이 아니라, **에이전트가 학습하고 추론할 수 있는 지식 기반(OKF KB)과 실행 환경(ActiveGraph)을 조성**하는 것입니다:

1. **온톨로지와 규칙 정의**: 어떤 개념이 원자적인지, 어떤 관계가 유효한지 정의
2. **에이전트 제안 검토 및 승인**: 에이전트가 제안한 개선안을 검토하고 승인/거부
3. **지식 그래프 품질 관리**: OKF KB의 정확성, 완전성, 일관성 유지
4. **학습 경로(curriculum) 설계**: 어떤 순서로 개념을 학습할지 정의

---

## 📖 관련 자료

- [ActiveGraph 공식 문서](https://activegraph.ai)
- [OKF Skills Repository](https://github.com/scaccogatto/okf-skills)
- [Regimes 논문 (arXiv:2606.10241)](https://arxiv.org/abs/2606.10241)
- [LongMemEval 벤치마크](https://github.com/example/longmemeval)

---

## 📄 라이선스

MIT License

---

*"The substrate is the lever; the graph is the controller; the agent is the learner; and the curator is the guide."*