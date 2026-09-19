"""
Resolve curriculum combo slots (SE_COM*, PHE_COM*) → concrete subject codes
based on specialty track inferred from career goal (or explicit track).

Scans knowledge_base/**/SE_COM*/README.md at runtime (no re-ingest required).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent / "data" / "knowledge_base"

# Goal keywords → folder/section name used in README headings
TRACK_ALIASES = {
    "java": "Java chuyên sâu",
    "spring": "Java chuyên sâu",
    "hsf": "Java chuyên sâu",
    ".net": "lập trình .NET",
    "dotnet": "lập trình .NET",
    "asp.net": "lập trình .NET",
    "c#": "lập trình .NET",
    "csharp": "lập trình .NET",
    "prn": "lập trình .NET",
    "ai": "AI",
    "machine learning": "AI",
    "deep learning": "AI",
    "trí tuệ nhân tạo": "AI",
    "data science": "Khoa học dữ liệu (KHDL) ứng dụng",
    "khdl": "Khoa học dữ liệu (KHDL) ứng dụng",
    "data": "Khoa học dữ liệu (KHDL) ứng dụng",
    "game": "Phát triển game",
    "react": "React NodeJS",
    "node": "React NodeJS",
    "nodejs": "React NodeJS",
    "frontend": "React NodeJS",
    "devops": "Tích hợp DevSepOps cho cloud",
    "devsecops": "Tích hợp DevSepOps cho cloud",
    "cloud": "Tích hợp DevSepOps cho cloud",
    "vi mạch": "Thiết kế vi mạch",
    "hardware": "Thiết kế vi mạch",
    "nhật": "Kỹ sư cầu nối Nhật Bản (Định hướng Tiếng Nhật nâng cao cho kỹ sư CNTT)",
    "japanese": "Kỹ sư cầu nối Nhật Bản (Định hướng Tiếng Nhật nâng cao cho kỹ sư CNTT)",
    "hàn": "Công nghệ thông tin - tiếng Hàn",
    "korean": "Công nghệ thông tin - tiếng Hàn",
}

PE_TRACK_ALIASES = {
    "vovinam": "Vovinam",
    "võ": "Vovinam",
    "cờ vua": "Cờ vua BIT",
    "chess": "Cờ vua BIT",
    "cov": "Cờ vua BIT",
}

# Curriculum: PHE_COM*N lives in early-semester SE_COM* folders in this vault
PHE_SLOT_TO_FOLDER = {
    "PHE_COM*1": (0, "SE_COM*1"),
    "PHE_COM*2": (1, "SE_COM*2"),
    "PHE_COM*3": (2, "SE_COM*3"),
}

# Specialty SE_COM slots by curriculum semester (primary)
SE_SLOT_SEMESTERS = {
    "SE_COM*1": [5, 7],  # Kỳ 5 specialty; Kỳ 7 also has SE_COM*1 (Korean etc.)
    "SE_COM*2": [7, 8, 1],
    "SE_COM*3": [7, 8, 2],
    "SE_COM*4_ELE": [8],
    "SE_COM*4": [8],
}

SLOT_CODE_RE = re.compile(
    r"^(PHE_COM\*?\d+|SE_COM\*?\d+(?:_ELE)?|SE_COM\d+(?:\.\d+)?(?:_\d+)?(?:_ELE)?)$",
    re.IGNORECASE,
)


def _normalize_slot_key(code: str) -> str | None:
    """Map messy LLM codes → canonical slot keys."""
    if not code:
        return None
    raw = code.strip().upper().replace(" ", "")
    raw = raw.replace("＊", "*")

    # Already canonical-ish
    m = re.match(r"^PHE_COM\*?(\d+)$", raw, re.I)
    if m:
        return f"PHE_COM*{m.group(1)}"

    m = re.match(r"^SE_COM\*?(\d+)(_ELE)?$", raw, re.I)
    if m:
        n, ele = m.group(1), m.group(2)
        if ele or n == "4":
            return "SE_COM*4_ELE"
        return f"SE_COM*{n}"

    # SE_COM10.2_4 / SE_COM3.3_2 style → slot number after last _
    m = re.match(r"^SE_COM[\d.]+_(\d+)(_ELE)?$", raw, re.I)
    if m:
        n = m.group(1)
        if n == "4" or m.group(2):
            return "SE_COM*4_ELE"
        return f"SE_COM*{n}"

    if "SE_COM" in raw and "4" in raw and "ELE" in raw:
        return "SE_COM*4_ELE"

    return None


def infer_specialty_track(goal: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    g = (goal or "").lower()
    # Prefer longer / more specific keys first
    for key in sorted(TRACK_ALIASES.keys(), key=len, reverse=True):
        if key in g:
            return TRACK_ALIASES[key]
    return "Java chuyên sâu"  # safe default for SE


def infer_pe_track(goal: str, explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    g = (goal or "").lower()
    for key, track in PE_TRACK_ALIASES.items():
        if key in g:
            return track
    return "Vovinam"


def _parse_readme_tracks(text: str) -> dict[str, tuple[str, str]]:
    """
    Parse README sections:
      ## Track Name (id)
      - [[CODE]] — Title
    → { track_name: (code, title) }
    """
    tracks: dict[str, tuple[str, str]] = {}
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s+(.+?)(?:\s*\(\d+\))?\s*$", line.strip())
        if h:
            current = h.group(1).strip()
            # Strip trailing (PHE_COMx) annotations
            current = re.sub(r"\s*\(PHE_COM\d+\)\s*$", "", current, flags=re.I).strip()
            continue
        if current is None:
            continue
        link = re.search(r"\[\[([^\]]+)\]\]\s*(?:—|-)?\s*(.*)$", line)
        if link:
            code = link.group(1).strip()
            title = (link.group(2) or code).strip() or code
            tracks[current] = (code, title)
    return tracks


def _semester_from_path(path: Path) -> int | None:
    for part in path.parts:
        m = re.match(r"Kỳ\s*(\d+)", part, re.I)
        if m:
            return int(m.group(1))
    return None


def _slot_from_dirname(name: str) -> str | None:
    cleaned = name.replace("＊", "*")
    m = re.match(r"(SE_COM\*\d+(?:_ELE)?)", cleaned, re.I)
    if m:
        key = m.group(1).upper().replace("SE_COM*", "SE_COM*")
        # Normalize casing of ELE
        key = re.sub(r"se_com", "SE_COM", key, flags=re.I)
        if key.upper().endswith("_ELE"):
            return "SE_COM*4_ELE"
        m2 = re.match(r"SE_COM\*(\d+)", key, re.I)
        if m2:
            return f"SE_COM*{m2.group(1)}"
    return None


@lru_cache(maxsize=1)
def load_combo_index() -> dict:
    """
    index[(semester, slot)][track] = (code, name)
    Also index_by_slot[slot][track] = list of (semester, code, name)
    """
    by_sem_slot: dict[tuple[int, str], dict[str, tuple[str, str]]] = {}
    if not KB_ROOT.exists():
        return {"by_sem_slot": by_sem_slot}

    for readme in KB_ROOT.rglob("README.md"):
        parent = readme.parent.name
        slot = _slot_from_dirname(parent)
        if not slot:
            continue
        sem = _semester_from_path(readme)
        if sem is None:
            continue
        try:
            text = readme.read_text(encoding="utf-8")
        except OSError:
            continue
        tracks = _parse_readme_tracks(text)
        if not tracks:
            continue
        by_sem_slot[(sem, slot)] = tracks

    return {"by_sem_slot": by_sem_slot}


def _pick_track_course(
    tracks: dict[str, tuple[str, str]],
    preferred: str,
    *,
    allow_fallback: bool = False,
) -> tuple[str, str, str] | None:
    """Return (code, name, matched_track). Prefer exact/alias match — no weak token fallback."""
    if not tracks:
        return None
    pref = preferred.lower().strip()

    # 1) Exact (case-insensitive)
    for name, (code, title) in tracks.items():
        if name.lower() == pref:
            return code, title, name

    # 2) Preferred contained in track name (or reverse) — only if preferred is long enough
    if len(pref) >= 4:
        for name, (code, title) in tracks.items():
            nl = name.lower()
            if pref in nl or (len(nl) >= 4 and nl in pref):
                return code, title, name

    # 3) Strong keyword aliases only (never single-letter / broken Unicode tokens)
    keyword_groups = [
        {"java", "spring", "hsf", "sba", "mss"},
        {".net", "dotnet", "prn", "pru", "asp.net", "c#", "csharp"},
        {"ai", "machine", "learning", "prp", "ail", "dbm", "dpl", "deep"},
        {"data", "khdl", "pds", "dhv", "mds", "bdt", "science"},
        {"game", "fgu", "agu", "gdc", "gns"},
        {"react", "node", "nodejs", "fer", "mma", "sdn", "wdp"},
        {"devops", "devsecops", "cloud", "iao", "prc", "asp", "dso"},
        {"vi mạch", "electronic", "circuit", "eci", "mip", "dcd", "acd"},
        {"nhật", "japanese", "jpd", "jfe", "jis", "jit"},
        {"hàn", "korean", "kor"},
        {"vovinam", "vov"},
        {"cờ vua", "chess", "cov"},
    ]

    pref_keys = set()
    for group in keyword_groups:
        if any(k in pref for k in group):
            pref_keys |= group

    if pref_keys:
        best = None
        best_score = 0
        for name, (code, title) in tracks.items():
            blob = f"{name} {code} {title}".lower()
            score = sum(1 for k in pref_keys if k in blob)
            if score > best_score:
                best_score = score
                best = (code, title, name)
        if best and best_score > 0:
            return best

    if allow_fallback:
        name = next(iter(tracks))
        code, title = tracks[name]
        return code, title, name
    return None


def resolve_slot(
    slot_key: str,
    semester: int | None,
    specialty_track: str,
    pe_track: str,
) -> dict | None:
    """
    Resolve one slot → {code, name, type, note, track}.
    """
    index = load_combo_index()["by_sem_slot"]

    if slot_key.startswith("PHE_COM"):
        mapped = PHE_SLOT_TO_FOLDER.get(slot_key)
        if not mapped:
            return None
        pe_sem, folder_slot = mapped
        tracks = index.get((pe_sem, folder_slot)) or {}
        # Also try alternate folder names in same semester
        if not tracks:
            for (s, sl), tr in index.items():
                if s == pe_sem and sl.startswith("SE_COM"):
                    if any(
                        "vovinam" in k.lower()
                        or "cờ" in k.lower()
                        or "cov" in k.lower()
                        for k in tr
                    ):
                        tracks = tr
                        break
        # PE: allow fallback to first option (Vovinam/COV)
        picked = _pick_track_course(tracks, pe_track, allow_fallback=True)
        if not picked:
            return None
        code, name, track = picked
        return {
            "code": code,
            "name": name,
            "type": "elective",
            "note": f"PHE {slot_key} → {track} ({code})",
            "track": track,
        }

    # Specialty SE_COM* — never fall back to unrelated track
    candidates_sem = []
    if semester is not None:
        candidates_sem.append(semester)
    candidates_sem.extend(SE_SLOT_SEMESTERS.get(slot_key, []))
    # Also search every semester that has this slot
    for (s, sl) in index.keys():
        if sl == slot_key or (slot_key == "SE_COM*4" and sl == "SE_COM*4_ELE"):
            candidates_sem.append(s)

    seen = set()
    sem_list = []
    for s in candidates_sem:
        if s not in seen:
            seen.add(s)
            sem_list.append(s)

    for sem in sem_list:
        lookup_slots = [slot_key]
        if slot_key == "SE_COM*4":
            lookup_slots.append("SE_COM*4_ELE")
        for sl in lookup_slots:
            tracks = index.get((sem, sl))
            if not tracks:
                continue
            pe_only = all(
                "vovinam" in k.lower() or "cờ" in k.lower() or "cov" in k.lower()
                for k in tracks
            )
            if pe_only:
                continue
            picked = _pick_track_course(
                tracks, specialty_track, allow_fallback=False
            )
            if not picked:
                continue
            code, name, track = picked
            return {
                "code": code,
                "name": name,
                "type": "combo",
                "note": f"{slot_key} · {track} → {code}",
                "track": track,
            }
    return None


def is_placeholder_code(code: str) -> bool:
    if not code:
        return False
    c = code.strip().upper().replace("＊", "*")
    if _normalize_slot_key(c):
        return True
    if re.search(r"SE_COM|PHE_COM", c):
        return True
    return False


def _is_pe_only_tracks(tracks: dict) -> bool:
    if not tracks:
        return True
    return all(
        "vovinam" in k.lower() or "cờ" in k.lower() or "cov" in k.lower()
        for k in tracks
    )


def build_specialty_path(specialty_track: str) -> list[dict]:
    """
    Danh sách môn chuyên ngành hẹp theo đúng kỳ thư mục KB.
    VD Java: Kỳ5 HSF302, Kỳ7 SBA301, Kỳ8 MSS301 — không gộp MSS vào Kỳ 7.
    """
    index = load_combo_index()["by_sem_slot"]
    path: list[dict] = []
    seen_codes: set[str] = set()
    for (sem, slot), tracks in sorted(index.items(), key=lambda x: (x[0][0], x[0][1])):
        if _is_pe_only_tracks(tracks):
            continue
        if not slot.startswith("SE_COM"):
            continue
        picked = _pick_track_course(tracks, specialty_track, allow_fallback=False)
        if not picked:
            continue
        code, name, track = picked
        code_key = code.upper()
        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)
        path.append(
            {
                "semester": sem,
                "slot": slot,
                "code": code,
                "name": name,
                "credits": 3,
                "type": "combo",
                "note": f"{slot} · {track} → {code} (Kỳ {sem})",
                "track": track,
            }
        )
    return path


def all_specialty_codes_in_kb() -> set[str]:
    """Mọi mã môn thuộc bất kỳ chuyên ngành hẹp nào (không gồm GDTC)."""
    index = load_combo_index()["by_sem_slot"]
    codes: set[str] = set()
    for (_, slot), tracks in index.items():
        if not slot.startswith("SE_COM"):
            continue
        if _is_pe_only_tracks(tracks):
            continue
        for _track, (code, _title) in tracks.items():
            if code:
                codes.add(code.upper())
    return codes


def build_pe_path(pe_track: str) -> list[dict]:
    """PHE_COM*1/2/3 → mã GDTC cụ thể đúng kỳ 0/1/2."""
    path = []
    for slot, (sem, _folder) in sorted(
        PHE_SLOT_TO_FOLDER.items(), key=lambda x: x[1][0]
    ):
        resolved = resolve_slot(slot, sem, "", pe_track)
        if resolved:
            path.append(
                {
                    "semester": sem,
                    "slot": slot,
                    "code": resolved["code"],
                    "name": resolved["name"],
                    "credits": 2,
                    "type": "elective",
                    "note": resolved["note"],
                }
            )
    return path


def expand_study_plan_combos(
    plan: dict,
    goal: str = "",
    combo_track: str | None = None,
    pe_track: str | None = None,
) -> dict:
    """
    1) Bỏ placeholder SE_COM*/PHE_COM*
    2) Bỏ môn combo lệch track (VD PRN* khi chọn Java)
    3) Gắn đúng môn chuyên ngành theo kỳ KB (HSF@5, SBA@7, MSS@8)
    4) Gắn GDTC cụ thể (VOV/COV) vào kỳ 0/1/2
    """
    if not plan or not isinstance(plan, dict):
        return plan

    specialty = infer_specialty_track(goal or plan.get("goal") or "", combo_track)
    pe = infer_pe_track(goal or plan.get("goal") or "", pe_track)

    specialty_path = build_specialty_path(specialty)
    pe_path = build_pe_path(pe)
    allowed_combo = {c["code"].upper() for c in specialty_path}
    foreign_combo = all_specialty_codes_in_kb() - allowed_combo

    plan = dict(plan)
    plan["combo_track"] = specialty
    plan["pe_track"] = pe
    warnings: list[str] = list(plan.get("warnings") or [])

    # Gom semester dict theo số kỳ
    by_sem: dict[int, dict] = {}
    for sem in plan.get("semesters") or []:
        if not isinstance(sem, dict):
            continue
        try:
            sem_num = int(sem.get("semester"))
        except (TypeError, ValueError):
            continue
        by_sem[sem_num] = {
            "semester": sem_num,
            "label": str(sem.get("label") or f"Kỳ {sem_num}"),
            "courses": [],
        }
        for c in sem.get("courses") or []:
            if not isinstance(c, dict):
                continue
            code = str(c.get("code") or "").strip()
            if not code:
                continue
            # Bỏ placeholder — sẽ inject từ path
            if is_placeholder_code(code):
                continue
            code_u = code.upper()
            # Bỏ môn chuyên ngành hẹp của track khác (PRN khi Java, HSF khi .NET…)
            if code_u in foreign_combo:
                warnings.append(
                    f"Đã loại {code} (không thuộc chuyên ngành {specialty})."
                )
                continue
            ctype = str(c.get("type") or "").lower()
            if ctype == "combo" and code_u not in allowed_combo:
                warnings.append(
                    f"Đã loại {code} (combo không khớp track {specialty})."
                )
                continue
            by_sem[sem_num]["courses"].append(c)

    def _ensure_sem(sem_num: int) -> dict:
        if sem_num not in by_sem:
            by_sem[sem_num] = {
                "semester": sem_num,
                "label": f"Kỳ {sem_num}",
                "courses": [],
            }
        return by_sem[sem_num]

    def _upsert_course(sem_num: int, course: dict) -> None:
        bucket = _ensure_sem(sem_num)
        code_u = course["code"].upper()
        # Xóa bản trùng / sai kỳ của cùng mã
        for s in by_sem.values():
            s["courses"] = [
                x
                for x in s["courses"]
                if str(x.get("code") or "").upper() != code_u
            ]
        bucket["courses"].append(
            {
                "code": course["code"],
                "name": course["name"],
                "credits": course.get("credits"),
                "type": course.get("type") or "combo",
                "note": course.get("note") or "",
            }
        )

    # Inject môn chuyên ngành đúng kỳ KB
    for item in specialty_path:
        _upsert_course(item["semester"], item)

    # Inject GDTC đúng kỳ 0/1/2 (chỉ nếu kỳ đó nằm trong lộ trình hiện tại
    # hoặc kỳ >= current — vẫn gắn nếu semester đã có trong plan / trong range)
    current = int(plan.get("current_semester") or 0)
    max_sem = max(by_sem.keys()) if by_sem else 9
    for item in pe_path:
        if item["semester"] < current:
            continue
        if item["semester"] > max_sem and item["semester"] not in by_sem:
            # vẫn thêm nếu nằm trong range lộ trình thông thường
            if item["semester"] > 9:
                continue
        _upsert_course(item["semester"], item)

    if not specialty_path:
        warnings.append(
            f"Không tìm thấy môn nào cho chuyên ngành '{specialty}' trong KB."
        )

    plan["warnings"] = warnings
    plan["semesters"] = [by_sem[k] for k in sorted(by_sem.keys())]
    return plan


def build_full_combo_catalog_text() -> str:
    """
    Catalog đầy đủ mọi combo SE + PHE kèm mã môn từng kỳ (từ KB README).
    Dùng inject vào prompt khi sinh viên hỏi liệt kê môn từng combo.
    """
    index = load_combo_index()["by_sem_slot"]
    # track_name -> list of (sem, slot, code, title)
    se_tracks: dict[str, list] = {}
    phe_tracks: dict[str, list] = {}

    for (sem, slot), tracks in sorted(index.items(), key=lambda x: (x[0][0], x[0][1])):
        is_pe = _is_pe_only_tracks(tracks)
        target = phe_tracks if is_pe else se_tracks
        for track_name, (code, title) in tracks.items():
            # Chuẩn hóa tên PHE (bỏ annotation)
            clean_name = re.sub(
                r"\s*\(PHE_COM\d+\)\s*$", "", track_name, flags=re.I
            ).strip()
            bucket = target.setdefault(clean_name, [])
            # Tránh trùng mã trong cùng track
            if any(c[2].upper() == code.upper() for c in bucket):
                continue
            bucket.append((sem, slot, code, title))

    lines = [
        "=== CATALOG COMBO ĐẦY ĐỦ (BẮT BUỘC dùng để liệt kê mã môn) ===",
        f"Tổng số combo: {len(se_tracks) + len(phe_tracks)} "
        f"({len(se_tracks)} chuyên ngành hẹp SE + {len(phe_tracks)} PHE).",
        "",
        "## A. Combo chuyên ngành hẹp (SE)",
    ]
    for i, name in enumerate(sorted(se_tracks.keys()), 1):
        lines.append(f"\n### {i}. {name}")
        for sem, slot, code, title in sorted(se_tracks[name], key=lambda x: (x[0], x[1])):
            lines.append(f"- Kỳ {sem} ({slot}): `{code}` — {title}")

    lines.append("\n## B. Combo Giáo dục thể chất (PHE)")
    for i, name in enumerate(sorted(phe_tracks.keys()), 1):
        lines.append(f"\n### {i}. {name}")
        for sem, slot, code, title in sorted(phe_tracks[name], key=lambda x: (x[0], x[1])):
            # Map SE_COM early folders → PHE slot label for clarity
            phe_slot = {
                0: "PHE_COM*1",
                1: "PHE_COM*2",
                2: "PHE_COM*3",
            }.get(sem, slot)
            lines.append(f"- Kỳ {sem} ({phe_slot}): `{code}` — {title}")

    lines.append(
        "\n(Nguồn: Combo Management.md + README slot SE_COM*/PHE trong knowledge base)"
    )
    return "\n".join(lines)


def build_combo_hint_for_prompt(goal: str, combo_track: str | None = None) -> str:
    """Short table injected into study_plan prompt so LLM emits concrete codes."""
    specialty = infer_specialty_track(goal, combo_track)
    pe = infer_pe_track(goal, None)
    lines = [
        f"Chuyên ngành hẹp (combo SE) đã chọn: **{specialty}**",
        f"Giáo dục thể chất (PHE): **{pe}**",
        "BẮT BUỘC: KHÔNG được để code dạng SE_COM*, PHE_COM*, SE_COM10.2_4.",
        "Phải thay bằng mã môn cụ thể ĐÚNG KỲ theo bảng (không gộp sai kỳ):",
    ]
    for item in build_specialty_path(specialty):
        lines.append(
            f"- Kỳ {item['semester']} / {item['slot']} → {item['code']} ({item['name']})"
        )
    for item in build_pe_path(pe):
        lines.append(
            f"- Kỳ {item['semester']} / {item['slot']} → {item['code']} ({item['name']})"
        )
    lines.append(
        f"KHÔNG được thêm môn chuyên ngành hẹp khác (VD không thêm PRN* khi track là {specialty})."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_combo_hint_for_prompt("Java Engineer"))
    print("---")
    print(build_combo_hint_for_prompt(".NET Developer"))
    sample = {
        "current_semester": 1,
        "goal": "JAVA Engineer",
        "semesters": [
            {
                "semester": 1,
                "label": "Kỳ 1",
                "courses": [
                    {"code": "PHE_COM*2", "name": "GDTC 2", "type": "elective"},
                    {"code": "PRF192", "name": "PRF", "type": "core"},
                ],
            },
            {
                "semester": 5,
                "label": "Kỳ 5",
                "courses": [
                    {"code": "SE_COM*1", "name": "Combo 1", "type": "combo"},
                ],
            },
            {
                "semester": 8,
                "label": "Kỳ 8",
                "courses": [
                    {"code": "SE_COM10.2_4", "name": "Combo 4", "type": "combo"},
                ],
            },
        ],
    }
    print(expand_study_plan_combos(sample, goal="JAVA Engineer"))
