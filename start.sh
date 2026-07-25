#!/usr/bin/env bash
# smart-cs 一键启动脚本
# 用法:
#   ./start.sh          # 启动 后端 + 管理后台 + 客户端
#   ./start.sh stop     # 停止所有服务
#   ./start.sh restart  # 重启
#   ./start.sh status   # 查看运行状态
#
# 注意: Windows 请在 Git Bash 中运行; 首次需先按 README 创建 venv 并安装依赖。

set -euo pipefail

# 定位到脚本所在目录 (smart-cs)
cd "$(dirname "$0")"

VENV_PY="venv/Scripts/python.exe"
BACKEND_PORT=8000
ADMIN_PORT=8501
CUSTOMER_PORT=8502

PID_DIR=".pids"
mkdir -p "$PID_DIR"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
ADMIN_PID_FILE="$PID_DIR/admin.pid"
CUSTOMER_PID_FILE="$PID_DIR/customer.pid"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[smart-cs]${NC} $*"; }
warn() { echo -e "${YELLOW}[smart-cs]${NC} $*"; }
err()  { echo -e "${RED}[smart-cs]${NC} $*"; }

check_venv() {
  if [ ! -x "$VENV_PY" ]; then
    err "未找到虚拟环境: $VENV_PY"
    err "请先创建 venv 并安装依赖, 参见 README.md:"
    err "  python -m venv venv"
    err "  venv\\\\Scripts\\\\pip install -r requirements.txt"
    exit 1
  fi
}

wait_for_health() {
  local port=$1 tries=40
  for ((i=1; i<=tries; i++)); do
    if curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_backend() {
  log "启动后端 (uvicorn, 端口 $BACKEND_PORT) ..."
  nohup "$VENV_PY" -m uvicorn api.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --workers 1 > uvicorn.log 2>&1 &
  echo $! > "$BACKEND_PID_FILE"
  if wait_for_health "$BACKEND_PORT"; then
    log "后端已就绪: http://127.0.0.1:$BACKEND_PORT  (API 文档 /docs)"
  else
    err "后端启动超时, 请查看 uvicorn.log"
    exit 1
  fi
}

start_frontend() {
  local name=$1 app=$2 port=$3 pidfile=$4
  log "启动 $name (streamlit, 端口 $port) ..."
  nohup "$VENV_PY" -m streamlit run "$app" --server.port "$port" --server.headless true --server.address 127.0.0.1 > "$name.log" 2>&1 &
  echo $! > "$pidfile"
  sleep 3
  log "$name 已启动: http://127.0.0.1:$port"
}

stop_service() {
  local name=$1 pidfile=$2
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile" 2>/dev/null || true)
    if [ -n "$pid" ]; then
      taskkill /F /PID "$pid" >/dev/null 2>&1 || kill -9 "$pid" 2>/dev/null || true
      warn "已停止 $name (pid $pid)"
    fi
    rm -f "$pidfile"
  fi
}

cmd_stop() {
  log "停止所有服务 ..."
  stop_service "backend"  "$BACKEND_PID_FILE"
  stop_service "admin"    "$ADMIN_PID_FILE"
  stop_service "customer" "$CUSTOMER_PID_FILE"
  pkill -f "uvicorn api.main"       2>/dev/null || true
  pkill -f "streamlit run admin_app"    2>/dev/null || true
  pkill -f "streamlit run customer_app" 2>/dev/null || true
  log "全部已停止"
}

cmd_status() {
  for entry in "backend:$BACKEND_PORT" "admin:$ADMIN_PORT" "customer:$CUSTOMER_PORT"; do
    name=${entry%%:*}; port=${entry##*:}
    if curl -s -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || \
       ( [ "$name" != "backend" ] && curl -s -m 2 "http://127.0.0.1:$port" >/dev/null 2>&1 ); then
      log "$name 运行中 (:$port)"
    else
      warn "$name 未运行 (:$port)"
    fi
  done
}

cmd_start() {
  check_venv
  if [ -f "$BACKEND_PID_FILE" ] || [ -f "$ADMIN_PID_FILE" ] || [ -f "$CUSTOMER_PID_FILE" ]; then
    warn "检测到已有服务在运行, 请先执行: ./start.sh stop"
    exit 1
  fi
  start_backend
  start_frontend "admin(管理后台-喂数据)"    "admin_app.py"    "$ADMIN_PORT"    "$ADMIN_PID_FILE"
  start_frontend "customer(客户端-问答)"     "customer_app.py" "$CUSTOMER_PORT" "$CUSTOMER_PID_FILE"
  echo
  log "===== 全部启动完成 ====="
  log "管理后台(喂数据): http://127.0.0.1:$ADMIN_PORT"
  log "客户端(问答)    : http://127.0.0.1:$CUSTOMER_PORT"
  log "后端 API        : http://127.0.0.1:$BACKEND_PORT  (文档 /docs)"
  echo
  warn "后台服务需用 './start.sh stop' 停止, 直接关闭终端不会结束进程。"
}

case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; sleep 2; cmd_start ;;
  status)  cmd_status ;;
  *) echo "用法: $0 {start|stop|restart|status}"; exit 1 ;;
esac
