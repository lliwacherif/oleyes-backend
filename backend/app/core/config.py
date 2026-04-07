import os

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


SCALWAY_API_KEY = get_env("SCALWAY_API_KEY")
SCALWAY_BASE_URL = get_env(
    "SCALWAY_BASE_URL",
    "https://api.scaleway.ai/1b0ca8d6-b434-4200-9818-4f56c17232ff/v1",
)
SCALWAY_MODEL = get_env("SCALWAY_MODEL", "gpt-oss-120b")
SCALWAY_TIMEOUT = float(get_env("SCALWAY_TIMEOUT", "30"))
SCALWAY_ANALYSIS_MODEL = get_env(
    "SCALWAY_ANALYSIS_MODEL",
    "gpt-oss-120b",
)
SCALWAY_SYSTEM_PROMPT = get_env(
    "SCALWAY_SYSTEM_PROMPT",
    'You are an AI Security Supervisor monitoring a CCTV feed. '
    'RULES: 1) TH:EFT If a Person overlaps an Item and the Item disappears '
    'while the Person moves away = HIGH risk. '
    '2) AGGRESSION: If two Persons are Nearby and one has high speed or '
    'Erratic movement = HIGH risk. '
    '3) NORMAL: People moving slowly or standing near items = LOW risk. '
    'Respond ONLY with valid JSON: '
    '{"risk_score": 0-100, "risk_level": "LOW"|"MEDIUM"|"HIGH", '
    '"label": "2-5 word title", "explanation": "1 sentence max 20 words"}',
)
SCALWAY_ANALYSIS_MAX_TOKENS = int(get_env("SCALWAY_ANALYSIS_MAX_TOKENS", "300"))
SCALWAY_ANALYSIS_TEMPERATURE = float(get_env("SCALWAY_ANALYSIS_TEMPERATURE", "0.2"))
SCALWAY_ANALYSIS_TOP_P = float(get_env("SCALWAY_ANALYSIS_TOP_P", "0.8"))
SCALWAY_ANALYSIS_PRESENCE_PENALTY = float(
    get_env("SCALWAY_ANALYSIS_PRESENCE_PENALTY", "0.0")
)

# ── Supreme OLEYES (Pixtral VLM) ─────────────────────────────────
SCALWAY_VLM_MODEL = get_env("SCALWAY_VLM_MODEL", "pixtral-12b-2409")
SUPREME_FRAME_COUNT = int(get_env("SUPREME_FRAME_COUNT", "3"))
SUPREME_FRAME_INTERVAL = float(get_env("SUPREME_FRAME_INTERVAL", "1.0"))
SUPREME_VLM_MAX_TOKENS = int(get_env("SUPREME_VLM_MAX_TOKENS", "500"))
SUPREME_VLM_TEMPERATURE = float(get_env("SUPREME_VLM_TEMPERATURE", "0.2"))
SUPREME_COOLDOWN = float(get_env("SUPREME_COOLDOWN", "10.0"))

YOLO_MODEL = get_env("YOLO_MODEL", "yolo26n.pt")
YOLO_POSE_MODEL = get_env("YOLO_POSE_MODEL", "yolo26n-pose.pt")
YOLO_DEVICE = get_env("YOLO_DEVICE", "cpu")
YOLO_CONF = float(get_env("YOLO_CONF", "0.25"))
YOLO_MAX_DETECTIONS = int(get_env("YOLO_MAX_DETECTIONS", "100"))
YOLO_INCLUDE_FRAME = get_env("YOLO_INCLUDE_FRAME", "false").lower() == "true"
YOLO_PREVIEW_EVERY = int(get_env("YOLO_PREVIEW_EVERY", "30"))
YOLO_LOG_EVERY = int(get_env("YOLO_LOG_EVERY", "1"))
YOLO_STREAM_EVERY = int(get_env("YOLO_STREAM_EVERY", "4"))
YOLO_VID_STRIDE = int(get_env("YOLO_VID_STRIDE", "1"))
YOLO_STREAM_BUFFER = get_env("YOLO_STREAM_BUFFER", "false").lower() == "true"

RTSP_URLS = get_env("RTSP_URLS")

# ── RTMP / MediaMTX ──────────────────────────────────────────────────
RTMP_BASE_URL = get_env("RTMP_BASE_URL", "rtmp://localhost:1935/live")

# ── PostgreSQL ───────────────────────────────────────────────────────
DATABASE_URL = get_env(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/oleyes",
)

# ── JWT / Auth ───────────────────────────────────────────────────────
JWT_SECRET_KEY = get_env(
    "JWT_SECRET_KEY",
    "dev-secret-change-me-in-production",
)
JWT_ALGORITHM = get_env("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(get_env("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(get_env("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
