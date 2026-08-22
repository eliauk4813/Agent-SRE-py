# Agent-SRE

基于 FastAPI、LangChain/LangGraph、MCP、Milvus 和 Elasticsearch 构建的智能运维 Agent 项目，主要面向 SRE 场景，提供 RAG 知识库问答、日志/监控工具调用和 AIOps 故障诊断能力。

## 核心能力

- **RAG 问答**：支持 Markdown 文档上传、切分、质量检查、向量索引和知识检索增强回答。
- **混合检索**：结合 Milvus 语义检索与 Elasticsearch 关键词检索，并通过 RRF 融合排序。
- **AIOps Agent**：基于 LangGraph 实现 Plan-Execute-Replan 流程，支持诊断计划生成、工具调用和最终报告输出。
- **MCP 工具接入**：通过 MCP Server 暴露日志、监控等运维工具，供 Agent 按需调用。
- **日志查询工具**：支持通过 provider 模式切换 mock 日志和 Elasticsearch 日志查询，便于从演示环境平滑接入真实日志索引。

## 项目结构

```text
app/
  api/                 FastAPI 接口
  agent/aiops/         AIOps Agent 节点：planner / executor / replanner
  services/            RAG、检索、索引、评估等核心服务
  tools/               LangChain 本地工具
mcp_servers/
  cls_server.py        日志查询 MCP Server
  monitor_server.py    监控查询 MCP Server
  providers/           日志 provider：mock / Elasticsearch
static/                简单前端页面
```

## 主要接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 服务与依赖健康检查 |
| `POST /api/chat` | 普通 RAG 对话 |
| `POST /api/chat_stream` | SSE 流式对话 |
| `POST /api/upload` | 上传 Markdown 并建立索引 |
| `POST /api/aiops` | AIOps 流式故障诊断 |

## 日志查询工具

日志 MCP Server 提供 `search_service_logs` 工具。后端服务日志默认已由外部链路采集到 Elasticsearch，Agent-SRE 只负责查询日志索引并将结果交给 Agent 分析。

典型流程：

```text
后端服务日志 -> Elasticsearch -> MCP search_service_logs -> AIOps Agent -> 诊断报告
```

支持按服务名、时间范围、日志级别、关键词、`trace_id`、`request_id`、接口路径、HTTP 状态码和慢请求阈值检索日志。

相关环境变量：

```env
AIOPS_LOG_PROVIDER=mock
AIOPS_LOG_ES_URL=http://localhost:9200
AIOPS_LOG_INDEX_PATTERN=springboot-logs-*
AIOPS_DEFAULT_SERVICE_NAME=data-sync-service
AIOPS_DEFAULT_ENV=prod
```

## 本地运行

项目依赖 Python 3.11+，并需要可用的 DashScope API Key、Milvus、Elasticsearch 以及 MCP Server。

安装依赖：

```bash
pip install -e .
```

启动 MCP Server：

```bash
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py
```


## 配置说明

主要配置来自 `.env`：

- `DASHSCOPE_API_KEY`：大模型与 embedding 调用密钥。
- `MILVUS_HOST` / `MILVUS_PORT`：Milvus 地址。
- `ES_URL` / `ES_INDEX_NAME`：RAG 关键词索引使用的 Elasticsearch 配置。
- `MCP_CLS_URL` / `MCP_MONITOR_URL`：MCP Server 地址。
- `AIOPS_LOG_PROVIDER`：AIOps 日志 provider，支持 `mock` 和 `elasticsearch`。

## 当前定位

该项目是一个面向智能运维场景的 Agent 工程实现，重点展示 RAG 知识库、MCP 工具调用、日志检索和 AIOps 诊断流程。生产落地时还需要结合实际环境完善认证鉴权、部署编排、监控告警、日志采集链路和权限审计。
