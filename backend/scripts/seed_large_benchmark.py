"""
Seed 100,000 benchmark students directly into PostgreSQL for Phase 5 load testing.
Uses PostgreSQL COPY via psycopg for maximum throughput (100k rows in ~3 seconds).
"""

import argparse
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

import psycopg
from backend.app.services.priority_scoring import compute_priority_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Diya", "Ananya", "Ishaan", "Saanvi", "Kabir",
    "Meera", "Rohan", "Priya", "Arjun", "Neha", "Kiran", "Riya", "Aryan",
    "Dev", "Pooja", "Vikram", "Sneha", "Tanvi", "Siddharth", "Rahul", "Anjali"
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Iyer", "Nair", "Singh", "Gupta",
    "Desai", "Menon", "Joshi", "Kulkarni", "Chopra", "Bose", "Mehta", "Rao"
]
MENTORS = [f"FAC_{i:03d}" for i in range(1, 25)]


def generate_benchmark_cohort(n: int = 100_000, target_db_url: str = None):
    db_url = target_db_url or os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("No database URL provided or found in environment.")

    # Remove +psycopg prefix if present for raw psycopg.connect
    clean_url = db_url.replace("postgresql+psycopg://", "postgresql://").replace("postgres://", "postgresql://")

    logger.info("Connecting to PostgreSQL at %s ...", clean_url.split("@")[-1])
    t0 = time.perf_counter()

    with psycopg.connect(clean_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Check existing count
            cur.execute("SELECT count(*) FROM students;")
            existing = cur.fetchone()[0]
            logger.info("Found %d existing students in database.", existing)

            logger.info("Generating and copying %d benchmark student records...", n)
            now_dt = datetime.now(timezone.utc)
            base_features = {
                "age": 20.0,
                "commute_distance_km": 15.0,
                "income_slab_idx": 2,
                "is_first_generation": 0,
                "has_scholarship": 1,
                "fee_payment_delay_days": 10,
                "hostel_status": "Day Scholar",
                "att_core1": 80.0,
                "att_core2": 82.0,
                "att_lab": 85.0,
                "att_elective": 78.0,
                "attendance_month_1": 82.0,
                "attendance_month_2": 80.0,
                "attendance_month_3": 79.0,
                "attendance_percentage": 80.0,
                "attendance_3m_trend": -1.0,
                "consecutive_absences": 2,
                "attendance_risk_flag": 0,
                "prev_sem_cgpa": 7.5,
                "current_cgpa": 7.3,
                "cgpa_delta": -0.2,
                "backlog_count": 0,
                "internal_exam_score_pct": 72.0,
                "stem_core_fail_flag": 0,
                "lms_logins_per_week": 6.5,
                "assignment_submission_lag_days": 0.5,
                "resource_access_count": 45,
                "days_since_last_lms_activity": 3,
                "forum_participation_count": 4,
            }
            features_json = json.dumps(base_features)

            # We use COPY directly for maximum insertion speed
            logger.info("Streaming students table via COPY...")
            student_rows = []
            pred_rows = []
            latest_rows = []

            rng = random.Random(42)

            for i in range(1, n + 1):
                sid_uuid = uuid.uuid4()
                pred_uuid = uuid.uuid4()
                student_id = f"BENCH_{i:06d}"
                name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                dept = rng.choice(DEPARTMENTS)
                mentor = rng.choice(MENTORS)

                # Distribution: ~20% High, ~30% Medium, ~50% Low
                r = rng.random()
                if r < 0.20:
                    prob = round(rng.uniform(0.70, 0.98), 4)
                    tier = "High"
                    backlogs = rng.randint(2, 6)
                    att = round(rng.uniform(40.0, 74.0), 1)
                elif r < 0.50:
                    prob = round(rng.uniform(0.40, 0.6999), 4)
                    tier = "Medium"
                    backlogs = rng.randint(0, 2)
                    att = round(rng.uniform(75.0, 84.0), 1)
                else:
                    prob = round(rng.uniform(0.02, 0.3999), 4)
                    tier = "Low"
                    backlogs = 0
                    att = round(rng.uniform(85.0, 99.0), 1)

                priority = compute_priority_score(
                    calibrated_risk_probability=prob,
                    backlog_count=backlogs,
                    attendance_percentage=att,
                )

                student_rows.append((sid_uuid, student_id, name, dept, mentor, features_json, now_dt, now_dt))
                pred_rows.append((pred_uuid, sid_uuid, prob, tier, round(prob * 100.0, 2), "calibrated-v1", features_json, None, now_dt))
                latest_rows.append((sid_uuid, pred_uuid, prob, tier, priority, now_dt, dept, mentor))

            # COPY INTO students
            with cur.copy("COPY students (id, student_id, name, department, assigned_mentor_id, features, created_at, updated_at) FROM STDIN") as copy:
                for row in student_rows:
                    copy.write_row(row)

            # COPY INTO predictions
            logger.info("Streaming predictions table via COPY...")
            with cur.copy("COPY predictions (id, student_id, calibrated_risk_probability, risk_tier, risk_score_percentage, model_version, input_features, top_drivers, evaluated_at) FROM STDIN") as copy:
                for row in pred_rows:
                    copy.write_row(row)

            # COPY INTO latest_predictions
            logger.info("Streaming latest_predictions table via COPY...")
            with cur.copy("COPY latest_predictions (student_id, prediction_id, calibrated_risk_probability, risk_tier, priority_score, evaluated_at, department, assigned_mentor_id) FROM STDIN") as copy:
                for row in latest_rows:
                    copy.write_row(row)

            conn.commit()

            # Analyze table statistics for Postgres query planner
            logger.info("Running ANALYZE on modified tables...")
            cur.execute("ANALYZE students;")
            cur.execute("ANALYZE predictions;")
            cur.execute("ANALYZE latest_predictions;")
            conn.commit()

    elapsed = time.perf_counter() - t0
    logger.info("Seeded %d benchmark students in %.2f seconds (%.0f rows/sec)!", n, elapsed, n / elapsed)
    return n


def main():
    parser = argparse.ArgumentParser(description="Seed 100k students for load testing")
    parser.add_argument("--count", type=int, default=100_000, help="Number of students to seed")
    parser.add_argument("--db-url", type=str, default=None, help="Target PostgreSQL database URL")
    args = parser.parse_args()

    generate_benchmark_cohort(n=args.count, target_db_url=args.db_url)


if __name__ == "__main__":
    main()
