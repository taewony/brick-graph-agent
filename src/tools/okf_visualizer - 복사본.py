import os
import re
import json
import yaml
from pathlib import Path

def parse_okf_file(file_path):
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
                print(f"Error parsing YAML in {file_path}: {e}")

    # Convert non-serializable frontmatter objects (like datetime) to string
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj

    frontmatter = sanitize(frontmatter)

    # Extract markdown links [link text](relative_path)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body)
    
    return {
        'id': frontmatter.get('id', str(file_path.relative_to(file_path.parents[2])).replace('\\', '/')),
        'title': frontmatter.get('title', file_path.stem),
        'type': frontmatter.get('type', 'Document'),
        'path': str(file_path.relative_to(file_path.parents[2])).replace('\\', '/'),
        'frontmatter': frontmatter,
        'body': body,
        'links': links
    }

def build_okf_visualizer(okf_dir: Path, output_file: Path):
    nodes = []
    edges = []
    node_map = {}

    okf_dir = okf_dir.resolve()
    
    # 1. Scan all markdown files
    md_files = list(okf_dir.rglob("*.md"))
    
    for md_file in md_files:
        parsed = parse_okf_file(md_file)
        node_id = parsed['id']
        node_map[parsed['path']] = node_id
        
        # Color coding by type/folder
        group = 'other'
        if '01_atomic_concepts' in parsed['path']:
            group = 'atomic'
        elif '02_composite_concepts' in parsed['path']:
            group = 'composite'
        elif '03_modules' in parsed['path']:
            group = 'module'
        elif '00_meta' in parsed['path']:
            group = 'meta'

        nodes.append({
            'id': node_id,
            'title': parsed['title'],
            'type': parsed['type'],
            'path': parsed['path'],
            'group': group,
            'body': parsed['body'],
            'frontmatter': parsed['frontmatter']
        })

    # 2. Extract edges from links & frontmatter prerequisites/composes
    for node in nodes:
        src_id = node['id']
        body = node['body']
        fm = node['frontmatter']
        
        # Extract explicit MD links
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', body)
        for link_text, target_path in md_links:
            # Normalize path
            if target_path.startswith('file:///'):
                # Handle file:// URL
                clean_path = target_path.replace('file:///', '').replace('\\', '/')
                # Extract relative .okf part
                if '.okf/' in clean_path:
                    okf_rel_path = '.okf/' + clean_path.split('.okf/')[1]
                    if okf_rel_path in node_map:
                        edges.append({
                            'source': src_id,
                            'target': node_map[okf_rel_path],
                            'relation': 'REFERENCES'
                        })

    # Remove duplicates
    unique_edges = []
    seen_edges = set()
    for edge in edges:
        pair = (edge['source'], edge['target'], edge['relation'])
        if pair not in seen_edges and edge['source'] != edge['target']:
            seen_edges.add(pair)
            unique_edges.append(edge)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    html_content = generate_dashboard_html(nodes, unique_edges)
    output_file.write_text(html_content, encoding='utf-8')
    print(f"✅ Dashboard successfully created at: {output_file}")

def generate_dashboard_html(nodes, edges):
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrickGraphAgent - OKF 지식 그래프 대시보드</title>
    <!-- Vis.js Network for Graph Visualization -->
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <!-- Marked.js for Markdown Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js for Syntax Highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        /* Sidebar Navigation & Directory Tree */
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
            margin-bottom: 6px;
        }}
        .sidebar-header p {{
            font-size: 0.8rem;
            color: #94a3b8;
        }}
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
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .node-item:hover {{
            background-color: #475569;
            transform: translateX(4px);
        }}
        .node-item.active {{
            background-color: #0284c7;
            color: #ffffff;
        }}
        .badge {{
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
            font-weight: bold;
        }}
        .badge-atomic {{ background-color: #38bdf8; color: #0f172a; }}
        .badge-composite {{ background-color: #a855f7; color: #ffffff; }}
        .badge-module {{ background-color: #22c55e; color: #0f172a; }}
        .badge-meta {{ background-color: #eab308; color: #0f172a; }}
        
        /* Main View Container */
        #main-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        #graph-view {{
            height: 45%;
            background-color: #0f172a;
            border-bottom: 1px solid #334155;
            position: relative;
        }}
        #content-view {{
            height: 55%;
            background-color: #1e293b;
            padding: 30px;
            overflow-y: auto;
        }}
        
        /* Markdown Content Styling */
        .markdown-body {{
            max-width: 900px;
            margin: 0 auto;
            line-height: 1.7;
        }}
        .markdown-body h1 {{
            color: #38bdf8;
            border-bottom: 1px solid #334155;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .markdown-body h2 {{
            color: #818cf8;
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        .markdown-body p {{
            margin-bottom: 14px;
            color: #cbd5e1;
        }}
        .markdown-body code {{
            background-color: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: #f43f5e;
            font-family: monospace;
        }}
        .markdown-body pre {{
            background-color: #0f172a;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 16px;
            border: 1px solid #334155;
        }}
        .markdown-body blockquote {{
            border-left: 4px solid #38bdf8;
            padding-left: 16px;
            color: #94a3b8;
            margin-bottom: 16px;
        }}
        .markdown-body a {{
            color: #38bdf8;
            text-decoration: none;
        }}
        .markdown-body a:hover {{
            text-decoration: underline;
        }}
        .meta-tags {{
            display: flex;
            gap: 10px;
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
        <div class="node-list" id="node-list">
            <!-- Node items will be populated by JS -->
        </div>
    </div>

    <div id="main-container">
        <div id="graph-view"></div>
        <div id="content-view">
            <div id="markdown-container" class="markdown-body">
                <h1>노드를 선택해 주세요</h1>
                <p>좌측 목록 또는 상단 그래프에서 노드를 클릭하면 마크다운 문서 내용을 실시간으로 브라우징할 수 있습니다.</p>
            </div>
        </div>
    </div>

    <script>
        const nodesData = {nodes_json};
        const edgesData = {edges_json};

        // Render Sidebar List
        const nodeListEl = document.getElementById('node-list');
        nodesData.forEach(node => {{
            const item = document.createElement('div');
            item.className = 'node-item';
            item.id = `sidebar-${{node.id}}`;
            item.innerHTML = `
                <span>${{node.title}}</span>
                <span class="badge badge-${{node.group}}">${{node.group}}</span>
            `;
            item.onclick = () => selectNode(node.id);
            nodeListEl.appendChild(item);
        }});

        // Vis.js Network Setup
        const visNodes = new vis.DataSet(nodesData.map(n => {{
            let color = '#38bdf8';
            if (n.group === 'composite') color = '#a855f7';
            if (n.group === 'module') color = '#22c55e';
            if (n.group === 'meta') color = '#eab308';
            return {{
                id: n.id,
                label: n.title,
                color: {{ background: color, border: '#ffffff' }},
                font: {{ color: '#ffffff' }}
            }};
        }}));

        const visEdges = new vis.DataSet(edgesData.map((e, idx) => ({{
            id: idx,
            from: e.source,
            to: e.target,
            arrows: 'to',
            color: {{ color: '#475569' }}
        }})));

        const container = document.getElementById('graph-view');
        const networkData = {{ nodes: visNodes, edges: visEdges }};
        const options = {{
            nodes: {{
                shape: 'dot',
                size: 16,
                font: {{ size: 14 }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{ gravitationalConstant: -3000, springLength: 95 }}
            }}
        }};
        const network = new vis.Network(container, networkData, options);

        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                selectNode(nodeId);
            }}
        }});

        function selectNode(nodeId) {{
            const node = nodesData.find(n => n.id === nodeId);
            if (!node) return;

            // Highlight Sidebar
            document.querySelectorAll('.node-item').forEach(el => el.classList.remove('active'));
            const activeEl = document.getElementById(`sidebar-${{nodeId}}`);
            if (activeEl) activeEl.classList.add('active');

            // Render Markdown Body
            const containerEl = document.getElementById('markdown-container');
            const parsedHtml = marked.parse(node.body);
            
            containerEl.innerHTML = `
                <div class="meta-tags">
                    <span class="meta-tag">ID: ${{node.id}}</span>
                    <span class="meta-tag">Type: ${{node.type}}</span>
                    <span class="meta-tag">Path: ${{node.path}}</span>
                </div>
                ${{parsedHtml}}
            `;

            // Apply syntax highlighting
            hljs.highlightAll();

            // Intercept Markdown Links to browse in-page
            containerEl.querySelectorAll('a').forEach(link => {{
                const href = link.getAttribute('href');
                if (href && (href.includes('.okf/') || href.includes('.md'))) {{
                    link.onclick = (e) => {{
                        e.preventDefault();
                        const targetNode = nodesData.find(n => href.includes(n.path) || href.includes(n.id));
                        if (targetNode) {{
                            selectNode(targetNode.id);
                            network.selectNodes([targetNode.id]);
                        }}
                    }};
                }}
            }});
        }}

        // Select Index by Default
        const indexNode = nodesData.find(n => n.id.includes('index') || n.group === 'meta') || nodesData[0];
        if (indexNode) {{
            selectNode(indexNode.id);
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OKF KB Visualizer & HTML Dashboard Generator")
    parser.add_argument("okf_dir", help="Path to .okf directory")
    parser.add_argument("-o", "--output", default="dist/index.html", help="Output HTML file path")
    args = parser.parse_args()
    
    build_okf_visualizer(Path(args.okf_dir), Path(args.output))
