"""临时离线检查运行器：仅跑不需要 API Key 的用例。"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "tests", "test_full.py")
spec = importlib.util.spec_from_file_location("mod_x", path)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

offline = [
    t.test_imports,
    t.test_namespace,
    t.test_agent_registry,
    t.test_qa_cache,
    t.test_excel_loader,
    t.test_build_agent_prompt,
    t.test_answer_cleaning,
    t.test_kb_service_and_routers,
]

for fn in offline:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        t.check(f"{fn.__name__} 异常: {e}", False)

r = t.RESULTS
print("\n" + "=" * 60)
print(f"离线检查结果: PASS={r['passed']}  FAIL={r['failed']}  WARN={r['skipped']}")
print("=" * 60)
sys.exit(1 if r["failed"] else 0)
