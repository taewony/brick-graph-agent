import os
import re
import json
import sys
import posixpath
import yaml
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj

def parse_okf_file(file_path, base_dir):
    content = file_path.read_text(encoding='utf-8')
    frontmatter = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()
            except Exception as e:
                print(f"⚠️ YAML 파싱 오류 ({file_path}): {e}")

    frontmatter = sanitize(frontmatter)

    rel_path = file_path.relative_to(base_dir)
    node_id = frontmatter.get('id')
    if not node_id:
        node_id = str(rel_path).replace('\\', '/')
        if node_id.endswith('.md'):
            node_id = node_id[:-3]

    group = 'other'
    path_str = str(rel_path).replace('\\', '/')
    if '01_atomic_concepts' in path_str or '/03_atomic/' in path_str or 'atomic' in path_str.lower():
        group = 'atomic'
    elif '02_composite_concepts' in path_str or '/02_composite/' in path_str or 'composite' in path_str.lower():
        group = 'composite'
    elif '03_modules' in path_str or '/01_module/' in path_str or 'module' in path_str.lower():
        group = 'module'
    elif '00_meta' in path_str or '/meta/' in path_str:
        group = 'meta'

    return {
        'id': node_id,
        'title': frontmatter.get('title', file_path.stem),
        'type': frontmatter.get('type', 'Document'),
        'path': str(rel_path).replace('\\', '/'),
        'group': group,
        'body': body,
        'frontmatter': frontmatter,
        'links': extract_links(body)
    }

def extract_links(body):
    return re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body)

def build_okf_visualizer(okf_dir: Path, output_file: Path):
    okf_dir = okf_dir.resolve()
    md_files = list(okf_dir.rglob("*.md"))
    nodes = []
    node_map = {}

    for md_file in md_files:
        node = parse_okf_file(md_file, okf_dir)
        nodes.append(node)
        node_map[node['path']] = node['id']
        if 'id' in node['frontmatter']:
            node_map[node['frontmatter']['id']] = node['id']

    edges = []
    for node in nodes:
        src_id = node['id']
        body = node['body']
        fm = node['frontmatter']

        for link_text, target in extract_links(body):
            if target.startswith('file:///'):
                clean = target.replace('file:///', '').replace('\\', '/')
                if '.okf/' in clean:
                    after = clean.split('.okf/', 1)[1]
                    # drop the bundle-name segment: .okf/01_nano_vllm/concepts/... -> concepts/...
                    seg = after.split('/', 1)
                    rel = seg[1] if len(seg) > 1 else seg[0]
                else:
                    rel = clean
            else:
                base_dir = Path(node['path']).parent
                rel = posixpath.normpath(str(base_dir / target).replace('\\', '/'))

            target_id = node_map.get(rel)
            if target_id and target_id != src_id:
                edges.append({'source': src_id, 'target': target_id, 'relation': 'REFERENCES'})

        for field in ['prerequisites', 'composes_into', 'prerequisite_of']:
            if field in fm:
                items = fm[field]
                if isinstance(items, str):
                    items = [items]
                for item in items:
                    if item in node_map:
                        target_id = node_map[item]
                        if target_id != src_id:
                            edges.append({'source': src_id, 'target': target_id, 'relation': field.upper()})

    unique_edges = []
    seen = set()
    for e in edges:
        key = (e['source'], e['target'], e['relation'])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    nodes_json = json.dumps(nodes, ensure_ascii=False, indent=2).replace('</script>', '<\\/script>')
    edges_json = json.dumps(unique_edges, ensure_ascii=False, indent=2).replace('</script>', '<\\/script>')

    html_content = generate_dashboard_html(nodes_json, edges_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')
    print(f"✅ 대시보드 생성 완료: {output_file}")

def generate_dashboard_html(nodes_json, edges_json):
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrickGraphAgent - OKF 지식 그래프 대시보드</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        #sidebar {{
            width: 320px;
            background-color: #1e293b;
            border-right: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }}
        .sidebar-header {{
            padding: 20px;
            background-color: #0f172a;
            border-bottom: 1px solid #334155;
        }}
        .sidebar-header h1 {{
            font-size: 1.2rem;
            color: #38bdf8;
        }}
        .sidebar-header p {{
            font-size: 0.8rem;
            color: #94a3b8;
        }}
        .nav-bar {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 15px;
            background-color: #0f172a;
            border-bottom: 1px solid #334155;
        }}
        .nav-btn {{
            padding: 4px 12px;
            border-radius: 6px;
            border: 1px solid #334155;
            background-color: #1e293b;
            color: #e2e8f0;
            font-size: 0.78rem;
            cursor: pointer;
            transition: 0.15s;
        }}
        .nav-btn:hover:not(:disabled) {{
            background-color: #0284c7;
            border-color: #0284c7;
        }}
        .nav-btn:disabled {{
            opacity: 0.35;
            cursor: default;
        }}
        .nav-pos {{
            margin-left: auto;
            font-size: 0.75rem;
            color: #94a3b8;
        }}
        .filter-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding: 10px 15px;
            background-color: #1e293b;
            border-bottom: 1px solid #334155;
        }}
        .filter-btn {{
            padding: 4px 12px;
            border-radius: 20px;
            border: 2px solid transparent;
            font-size: 0.75rem;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
            background-color: #334155;
            color: #cbd5e1;
        }}
        .filter-btn.active {{
            border-color: #ffffff;
            box-shadow: 0 0 8px rgba(255,255,255,0.3);
        }}
        .filter-btn.atomic {{ background-color: #38bdf8; color: #0f172a; }}
        .filter-btn.composite {{ background-color: #a855f7; color: #fff; }}
        .filter-btn.module {{ background-color: #22c55e; color: #0f172a; }}
        .filter-btn.meta {{ background-color: #eab308; color: #0f172a; }}
        .filter-btn.other {{ background-color: #64748b; color: #fff; }}
        .filter-btn.all {{ background-color: #475569; color: #fff; }}
        .filter-btn:hover {{ opacity: 0.8; transform: scale(1.05); }}
        .node-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}
        .node-item {{
            padding: 10px 14px;
            margin-bottom: 6px;
            background-color: #334155;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: 0.2s;
        }}
        .node-item:hover {{
            background-color: #475569;
            transform: translateX(4px);
        }}
        .node-item.active {{
            background-color: #0284c7;
            color: #fff;
        }}
        .badge {{
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .badge-atomic {{ background-color: #38bdf8; color: #0f172a; }}
        .badge-composite {{ background-color: #a855f7; color: #fff; }}
        .badge-module {{ background-color: #22c55e; color: #0f172a; }}
        .badge-meta {{ background-color: #eab308; color: #0f172a; }}
        .badge-other {{ background-color: #64748b; color: #fff; }}
        #main-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #graph-view {{
            height: 45%;          /* 초기 비율 */
            background-color: #0f172a;
            border-bottom: 1px solid #334155;
            overflow: hidden;
            min-height: 50px;     /* 너무 작아지지 않도록 */
            flex-shrink: 0;
        }}
        /* 분할 바 (드래그 핸들) */
        #resizer {{
            height: 6px;
            background-color: #334155;
            cursor: row-resize;
            flex-shrink: 0;
            transition: background-color 0.2s;
        }}
        #resizer:hover {{
            background-color: #38bdf8;
        }}
        #content-view {{
            flex: 1;              /* 남은 공간 모두 차지 */
            background-color: #1e293b;
            padding: 30px;
            overflow-y: auto;
            min-height: 100px;
        }}
        .markdown-body {{
            max-width: 900px;
            margin: 0 auto;
            line-height: 1.7;
        }}
        .markdown-body h1 {{ color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .markdown-body h2 {{ color: #818cf8; margin-top: 24px; }}
        .markdown-body p {{ color: #cbd5e1; margin-bottom: 14px; }}
        .markdown-body code {{
            background-color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: #f43f5e;
        }}
        .markdown-body pre {{
            background-color: #0f172a;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #334155;
            overflow-x: auto;
        }}
        .markdown-body blockquote {{
            border-left: 4px solid #38bdf8;
            padding-left: 16px;
            color: #94a3b8;
        }}
        .markdown-body a {{ color: #38bdf8; text-decoration: none; }}
        .markdown-body a:hover {{ text-decoration: underline; }}
        .meta-tags {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .meta-tag {{
            background-color: #334155;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header">
            <h1>🧱 BrickGraphAgent</h1>
            <p>OKF 지식 그래프 & 문서 브라우저</p>
        </div>
        <div class="nav-bar">
            <button id="btn-back" class="nav-btn" disabled title="뒤로 (Alt+←)">◀ 뒤로</button>
            <button id="btn-forward" class="nav-btn" disabled title="앞으로 (Alt+→)">앞으로 ▶</button>
            <span id="nav-position" class="nav-pos"></span>
        </div>
        <div class="filter-bar" id="filter-bar">
            <button class="filter-btn all active" data-group="all">전체</button>
            <button class="filter-btn atomic active" data-group="atomic">ATOMIC</button>
            <button class="filter-btn composite active" data-group="composite">COMPOSITE</button>
            <button class="filter-btn module active" data-group="module">MODULE</button>
            <button class="filter-btn meta active" data-group="meta">META</button>
            <button class="filter-btn other active" data-group="other">OTHER</button>
        </div>
        <div class="node-list" id="node-list"></div>
    </div>
    <div id="main-container">
        <div id="graph-view"></div>
        <div id="resizer"></div>   <!-- 드래그 핸들 -->
        <div id="content-view">
            <div id="markdown-container" class="markdown-body">
                <h1>노드를 선택해 주세요</h1>
                <p>좌측 목록 또는 그래프에서 노드를 클릭하면 내용이 표시됩니다.</p>
            </div>
        </div>
    </div>
    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const groupColors = {{
            atomic: '#38bdf8',
            composite: '#a855f7',
            module: '#22c55e',
            meta: '#eab308',
            other: '#64748b'
        }};

        let selectedGroups = new Set(['atomic', 'composite', 'module', 'meta', 'other']);

        const nodeListEl = document.getElementById('node-list');
        const graphContainer = document.getElementById('graph-view');
        const markdownContainer = document.getElementById('markdown-container');
        const resizer = document.getElementById('resizer');
        const contentView = document.getElementById('content-view');
        const mainContainer = document.getElementById('main-container');

        let network = null;

        // ---------- 드래그 리사이즈 로직 ----------
        let isResizing = false;
        let startY = 0;
        let startGraphHeight = 0;

        resizer.addEventListener('mousedown', function(e) {{
            isResizing = true;
            startY = e.clientY;
            startGraphHeight = graphContainer.offsetHeight;
            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';
        }});

        document.addEventListener('mousemove', function(e) {{
            if (!isResizing) return;
            const delta = e.clientY - startY;
            // delta가 양수면 그래프 영역을 줄임 (내림), 음수면 늘림 (올림)
            let newGraphHeight = startGraphHeight + delta;
            // 최소/최대 높이 제한 (전체 높이의 10% ~ 90%)
            const containerHeight = mainContainer.offsetHeight;
            const minH = containerHeight * 0.1;
            const maxH = containerHeight * 0.9;
            if (newGraphHeight < minH) newGraphHeight = minH;
            if (newGraphHeight > maxH) newGraphHeight = maxH;
            graphContainer.style.height = newGraphHeight + 'px';
        }});

        document.addEventListener('mouseup', function(e) {{
            if (isResizing) {{
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                // 그래프 크기 변경 후 네트워크 재조정 (필요시)
                if (network) {{
                    network.redraw();
                }}
            }}
        }});

        // 더블클릭하면 기본 비율(45%)로 복원
        resizer.addEventListener('dblclick', function() {{
            const containerHeight = mainContainer.offsetHeight;
            const defaultHeight = containerHeight * 0.45;
            graphContainer.style.height = defaultHeight + 'px';
            if (network) {{
                network.redraw();
            }}
        }});

        // ---------- 그래프 및 사이드바 렌더링 ----------
        function buildGraphData(groups) {{
            const filteredNodes = allNodes.filter(n => groups.has(n.group));
            const nodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredEdges = allEdges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
            return {{ nodes: filteredNodes, edges: filteredEdges }};
        }}

        function renderGraph(groups) {{
            const {{ nodes, edges }} = buildGraphData(groups);
            const visNodes = new vis.DataSet(nodes.map(n => ({{
                id: n.id,
                label: n.title,
                color: {{ background: groupColors[n.group] || '#64748b', border: '#ffffff' }},
                font: {{ color: '#ffffff' }}
            }})));
            const visEdges = new vis.DataSet(edges.map((e, idx) => ({{
                id: idx,
                from: e.source,
                to: e.target,
                arrows: 'to',
                color: {{ color: '#475569' }}
            }})));
            const data = {{ nodes: visNodes, edges: visEdges }};
            if (network === null) {{
                network = new vis.Network(graphContainer, data, {{
                    nodes: {{ shape: 'dot', size: 16, font: {{ size: 14 }} }},
                    physics: {{ stabilization: false, barnesHut: {{ gravitationalConstant: -3000, springLength: 95 }} }}
                }});
                network.on('click', function(params) {{
                    if (params.nodes.length) selectNode(params.nodes[0]);
                }});
            }} else {{
                network.setData(data);
                network.redraw();
            }}
        }}

        function renderSidebar(groups) {{
            const filtered = allNodes.filter(n => groups.has(n.group));
            nodeListEl.innerHTML = '';
            filtered.forEach(node => {{
                const item = document.createElement('div');
                item.className = 'node-item';
                item.id = 'sidebar-' + node.id;
                item.innerHTML = `<span>${{node.title}}</span><span class="badge badge-${{node.group}}">${{node.group}}</span>`;
                item.onclick = () => selectNode(node.id);
                nodeListEl.appendChild(item);
            }});
        }}

        function applyFilters() {{
            renderGraph(selectedGroups);
            renderSidebar(selectedGroups);
            // 현재 선택된 노드가 필터에 없으면 초기화
            const active = document.querySelector('.node-item.active');
            if (active) {{
                const id = active.id.replace('sidebar-', '');
                const node = allNodes.find(n => n.id === id);
                if (!node || !selectedGroups.has(node.group)) {{
                    document.querySelectorAll('.node-item').forEach(el => el.classList.remove('active'));
                    markdownContainer.innerHTML = `<h1>노드를 선택해 주세요</h1><p>좌측 목록 또는 그래프에서 노드를 클릭하면 내용이 표시됩니다.</p>`;
                }}
            }}
        }}

        // 필터 버튼 이벤트 (단일 선택)
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                const group = this.dataset.group;
                if (group === 'all') {{
                    selectedGroups = new Set(['atomic', 'composite', 'module', 'meta', 'other']);
                    document.querySelectorAll('.filter-btn:not(.all)').forEach(b => b.classList.add('active'));
                    this.classList.add('active');
                }} else {{
                    selectedGroups = new Set([group]);
                    document.querySelectorAll('.filter-btn:not(.all)').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    document.querySelector('.filter-btn.all').classList.remove('active');
                }}
                applyFilters();
            }});
        }});

        // 노드 선택 함수 — 브라우징 히스토리(앞으로/뒤로) 지원
        let history = [];
        let historyIndex = -1;
        const backBtn = document.getElementById('btn-back');
        const forwardBtn = document.getElementById('btn-forward');
        const positionEl = document.getElementById('nav-position');

        function updateNavButtons() {{
            backBtn.disabled = historyIndex <= 0;
            forwardBtn.disabled = historyIndex >= history.length - 1;
            positionEl.textContent = history.length > 0 ? (historyIndex + 1) + ' / ' + history.length : '';
        }}

        function renderNode(nodeId) {{
            const node = allNodes.find(n => n.id === nodeId);
            if (!node) return;
            document.querySelectorAll('.node-item').forEach(el => el.classList.remove('active'));
            const active = document.getElementById('sidebar-' + nodeId);
            if (active) active.classList.add('active');

            const parsed = marked.parse(node.body);
            markdownContainer.innerHTML = `
                <div class="meta-tags">
                    <span class="meta-tag">ID: ${{node.id}}</span>
                    <span class="meta-tag">Type: ${{node.type}}</span>
                    <span class="meta-tag">Path: ${{node.path}}</span>
                </div>
                ${{parsed}}
            `;
            hljs.highlightAll();

            // Resolve a link href relative to the DISPLAYED node's directory.
            // Module files link downward with file-relative paths like
            // "../03_atomic/kv_cache.md"; index links use "concepts/...".
            function resolveNodePath(href, baseNode) {{
                if (!href || /^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith('/') || href.startsWith('#')) {{
                    return href;
                }}
                const dir = baseNode.path.includes('/')
                    ? baseNode.path.split('/').slice(0, -1).join('/') : '';
                const joined = dir ? dir + '/' + href : href;
                const parts = [];
                for (const p of joined.split('/')) {{
                    if (p === '' || p === '.') continue;
                    if (p === '..') {{ if (parts.length) parts.pop(); }}
                    else parts.push(p);
                }}
                return parts.join('/');
            }}

            markdownContainer.querySelectorAll('a').forEach(link => {{
                const href = link.getAttribute('href');
                if (href && (href.includes('.okf/') || href.includes('.md'))) {{
                    link.onclick = (e) => {{
                        e.preventDefault();
                        const resolved = resolveNodePath(href, node);
                        const target =
                            allNodes.find(n => n.path === resolved)
                            || allNodes.find(n => resolved && (resolved.includes(n.path) || resolved.includes(n.id)))
                            || allNodes.find(n => href.includes(n.path) || href.includes(n.id));
                        if (target) {{
                            selectNode(target.id);
                            if (network) {{
                                network.selectNodes([target.id]);
                            }}
                        }}
                    }};
                }}
            }});
            updateNavButtons();
        }}

        // 기록(히스토리)을 남기며 노드 렌더링
        function selectNode(nodeId) {{
            const node = allNodes.find(n => n.id === nodeId);
            if (!node) return;
            if (history[historyIndex] === nodeId) {{
                renderNode(nodeId);
                return;
            }}
            // 현재 위치 이후의 앞으로-기록은 버리고 새 노드를 push
            history = history.slice(0, historyIndex + 1);
            history.push(nodeId);
            historyIndex = history.length - 1;
            renderNode(nodeId);
        }}

        function goBack() {{
            if (historyIndex > 0) {{
                historyIndex -= 1;
                renderNode(history[historyIndex]);
            }}
        }}

        function goForward() {{
            if (historyIndex < history.length - 1) {{
                historyIndex += 1;
                renderNode(history[historyIndex]);
            }}
        }}

        backBtn.addEventListener('click', goBack);
        forwardBtn.addEventListener('click', goForward);
        document.addEventListener('keydown', function(e) {{
            if (e.altKey && e.key === 'ArrowLeft') {{ e.preventDefault(); goBack(); }}
            if (e.altKey && e.key === 'ArrowRight') {{ e.preventDefault(); goForward(); }}
        }});

        // 초기 렌더링
        applyFilters();
        // 기본 선택
        const defaultNode = allNodes.find(n => n.id.includes('index') || n.group === 'meta') || allNodes[0];
        if (defaultNode) selectNode(defaultNode.id);

        // 윈도우 리사이즈 시 그래프 높이 조정 (기본 비율 유지)
        window.addEventListener('resize', function() {{
            // 사용자가 드래그로 높이를 변경했다면 그대로 두기 위해, 변경된 높이가 없으면 기본 비율로 재조정하지 않음.
            // 그냥 네트워크 redraw만 수행
            if (network) network.redraw();
        }});
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OKF KB Visualizer & HTML Dashboard Generator with Filtering")
    parser.add_argument("okf_dir", nargs="?", default="D:/code/brick-graph-agent/.okf", help="Path to .okf directory or subdirectory (e.g., .okf/01_nano_vllm)")
    parser.add_argument("-o", "--output", default=None, help="Output HTML file path (default: docs/<bundle_name>/index.html or docs/index.html)")
    args = parser.parse_args()

    okf_path = Path(args.okf_dir).resolve()

    if args.output:
        out_path = Path(args.output)
    else:
        if okf_path.name in (".okf", "okf"):
            out_path = Path("docs/index.html")
        else:
            out_path = Path(f"docs/{okf_path.name}/index.html")

    build_okf_visualizer(okf_path, out_path)