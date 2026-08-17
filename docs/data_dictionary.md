# DropoutGuard — Official Feature Data Dictionary
### Complete Specification of Input Features, Target Labels & Institutional Data Sources
**Project**: DropoutGuard (Smart India Hackathon 2026 PSID 7-L / SDG 4: Quality Education)  
**Schema Version**: 1.0 (Production Core)  
**Matrix Dimensions**: 2,000 students × 37 model input features (44 total columns)

---

## 1. Attendance & Classroom Discipline Pillar (10 Features)

Classroom attendance represents the primary regulatory and behavioral early warning signal in Indian higher education institutions (under mandatory AICTE/UGC guidelines).

| Feature Name | Display Title | Type | Scale / Range | Institutional Source | Definition & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `attendance_percentage` | Overall 3-Month Attendance | Float | `10.0 – 100.0%` | ERP Attendance System | Weighted 3-month attendance ($0.25 \times M_1 + 0.35 \times M_2 + 0.40 \times M_3$). Primary metric evaluated against the statutory 75% minimum threshold. |
| `attendance_month_1` | Month 1 Attendance | Float | `10.0 – 100.0%` | ERP Attendance System | Attendance percentage during the first 30 days of the semester (baseline engagement). |
| `attendance_month_2` | Month 2 Attendance | Float | `10.0 – 100.0%` | ERP Attendance System | Attendance percentage during the middle 30 days of the semester (mid-term engagement). |
| `attendance_month_3` | Current Month Attendance (M3) | Float | `10.0 – 100.0%` | ERP Attendance System | Attendance percentage during the most recent 30-day tracking window (acute disengagement). |
| `attendance_3m_trend` | 3-Month Attendance Trend Slope | Float | `-25.0 to +25.0%` | Derived Feature | Monthly rate of change in attendance: $(M_3 - M_1) / 2$. Negative values indicate progressive disengagement. |
| `consecutive_absences` | Consecutive Absent Days Streak | Integer | `0 – 30 days` | ERP Daily Swipe Logs | Longest continuous spell of unexcused absences. Spells $>10$ days indicate acute personal, health, or motivational crises. |
| `attendance_risk_flag` | Attendance Debarment Risk (<75%) | Binary | `{0, 1}` | Statutory Rule Engine | Set to `1` if `attendance_percentage < 75.0%`. Marks the student for mandatory parent alert and examination debarment review. |
| `att_core1` | Core Mathematics/Theory Course Attendance | Float | `10.0 – 100.0%` | Department Subject Register | Attendance in foundational mathematics and theoretical engineering sciences (e.g., Engineering Mathematics, Discrete Structures). |
| `att_core2` | Department Major Core Course Attendance | Float | `10.0 – 100.0%` | Department Subject Register | Attendance in primary departmental engineering courses (e.g., Data Structures, Signals & Systems, Thermodynamics). |
| `att_lab` | Practical Laboratory Course Attendance | Float | `15.0 – 100.0%` | Department Lab Register | Attendance in hands-on laboratory practicals. Practical shortfall directly blocks term-work submission. |
| `att_elective` | Elective Course Attendance | Float | `10.0 – 100.0%` | Department Subject Register | Attendance in departmental/open elective courses. Highlights elective-specific disinterest. |
| `subject_attendance_std` | Subject Attendance Variance | Float | `0.0 – 35.0` | Derived Feature | Standard deviation of attendance across all 4 subjects. High variance indicates selective class-cutting. |

---

## 2. Academic Performance & Semester Trajectory Pillar (7 Features)

Captures cumulative academic progression, semester velocity, and prerequisite course failure traps.

| Feature Name | Display Title | Type | Scale / Range | Institutional Source | Definition & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `current_cgpa` | Current Semester CGPA | Float | `0.00 – 10.00` | Examination Branch | Cumulative Grade Point Average on the standard Indian 10-point scale at the most recent evaluation checkpoint. |
| `prev_sem_cgpa` | Previous Semester CGPA | Float | `0.00 – 10.00` | Examination Branch | Historical cumulative GPA from the preceding academic semester (baseline reference). |
| `cgpa_delta` | Semester CGPA Trajectory | Float | `-3.50 to +3.50` | Derived Feature | Change in GPA: $\text{current\_cgpa} - \text{prev\_sem\_cgpa}$. Negative drops $>0.75$ indicate academic distress. |
| `backlog_count` | Uncleared Backlog Count | Integer | `0 – 8 subjects` | Examination Branch | Total active failed subjects. In Indian universities, $\ge 3$ backlogs triggers non-promotional "Year-Back" (academic detainment). |
| `internal_exam_score_pct` | Internal Assessment Marks | Float | `0.0 – 100.0%` | Faculty Gradebook | Continuous internal evaluation (In-Sem tests, quizzes, assignments). Leading mid-semester signal before final university end-sems. |
| `stem_core_fail_flag` | Core Course Failure Status | Binary | `{0, 1}` | Examination Branch | Set to `1` if the student failed prerequisite foundational STEM courses (e.g., Engineering Mathematics, Basic Electronics). |
| `academic_crisis_flag` | Academic Crisis Indicator | Binary | `{0, 1}` | Derived Indicator | Compound failure flag: set to `1` if `current_cgpa < 5.0` AND `backlog_count >= 2`. Identifies imminent academic detainment. |

---

## 3. Digital Learning Behavior & LMS Clickstream Pillar (6 Features)

Captures online engagement, study habits, and early digital withdrawal 2–4 weeks before physical classroom absenteeism.

| Feature Name | Display Title | Type | Scale / Range | Institutional Source | Definition & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lms_logins_per_week` | LMS Weekly Login Frequency | Float | `0.0 – 20.0 logins` | LMS Server Logs (Moodle/Canvas) | Average weekly authentication count on the institutional LMS portal. |
| `assignment_submission_lag_days` | Assignment Submission Delay | Float | `-5.0 to +15.0 days` | LMS Assignment Module | Average submission timing relative to deadline. Negative = early; positive = late/overdue. Proxy for academic conscientiousness. |
| `resource_access_count` | Digital Resource Access Count | Integer | `0 – 300 hits` | LMS Clickstream Logs | Total downloads of syllabus lecture slides, lab manuals, reference PDFs, and video lectures. |
| `days_since_last_lms_activity` | LMS Inactivity Recency | Integer | `0 – 60 days` | LMS Server Logs | Days since last user activity timestamp. Inactivity $>14$ days indicates silent digital withdrawal. |
| `forum_participation_count` | Discussion Forum Activity | Integer | `0 – 30 posts` | LMS Discussion Boards | Number of queries posted or peer responses contributed on course discussion forums. |
| `behavioral_disengagement_index` | Composite Behavioral Index | Float | `0.00 – 1.00` | Engineered Composite | Normalized composite index combining low logins, late submissions, and long inactivity: $\frac{1}{3}\left[\left(1 - \frac{\text{logins}}{12}\right) + \frac{\text{lag}}{10} + \frac{\text{inactivity}}{30}\right]$. |

---

## 4. Socio-Economic Resilience & Demographics Pillar (8 Features)

Captures non-academic institutional barriers, financial distress, and commuter friction.

| Feature Name | Display Title | Type | Scale / Range | Institutional Source | Definition & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `family_income_slab` | Family Income Bracket | Categorical | `<2 LPA, 2-5 LPA, 5-8 LPA, >8 LPA` | Admissions ERP | Annual family household income in Indian Lakhs Per Annum (LPA). |
| `income_slab_idx` | Income Bracket Level | Integer | `0, 1, 2, 3` | Encoded Categorical | Ordinal encoding (`0` = `<2 LPA` Economically Weaker Section, `3` = `>8 LPA` High Income). |
| `fee_payment_delay_days` | Tuition Fee Payment Overdue Days | Integer | `0 – 120 days` | Accounts & Fee Desk | Days tuition fee payment is delayed beyond the semester due date. Overdue $>45$ days indicates severe financial distress. |
| `has_scholarship` | Financial Scholarship Buffer | Binary | `{0, 1}` | Scholarship Desk / NSP | Indicates active government or institutional merit/means scholarship (e.g., Post-Matric, Pragati, PMSS). |
| `is_first_generation` | First-Generation College Learner | Binary | `{0, 1}` | Admissions ERP | Student whose parents have not completed higher education. Often lacks informal at-home academic mentorship. |
| `is_hosteler` | Hosteler Status | Binary | `{0, 1}` | Hostel Warden ERP | `1` = Campus Hostel Resident, `0` = Day Scholar. |
| `commute_distance_km` | Daily Commute Distance | Float | `0.5 – 50.0 km` | Admissions ERP | One-way daily travel distance for day scholars. Long commutes ($>20\text{ km}$) cause fatigue and morning absenteeism. |
| `financial_stress_index` | Composite Financial Stress Index | Float | `0.00 – 1.00` | Engineered Composite | Combines low income slab and fee payment delays: $\frac{1}{2}\left[\left(1 - \frac{\text{income\_idx}}{3}\right) + \min\left(1.0, \frac{\text{fee\_delay}}{60}\right)\right]$. |

---

## 5. Non-Linear Multi-Pillar Interaction Terms (4 Features)

Captures compounding systemic failure across multiple domains.

| Feature Name | Formula / Definition | Rationale |
| :--- | :--- | :--- |
| `interaction_att_x_fee` | $\max(0, (75 - \text{attendance})/10) \times (\text{fee\_delay}/30)$ | Dual-crisis indicator for students simultaneously facing exam debarment and tuition arrears. |
| `interaction_cgpa_x_backlog` | $\max(0, 6.5 - \text{current\_cgpa}) \times \text{backlog\_count}$ | Compound academic debt spiral where low GPA accelerates backlog accumulation. |
| `interaction_firstgen_x_inactivity` | $\text{is\_first\_generation} \times (\text{days\_since\_last\_lms\_activity}/14)$ | Captures silent digital withdrawal among first-generation learners lacking campus support. |
| `interaction_att_x_cgpa_drop` | $\max(0, (75 - \text{attendance})/10) \times \max(0, -\text{cgpa\_delta})$ | Synchronous collapse across classroom attendance and semester exam marks. |

---

## 6. Identifiers & Target Labels (3 Features)

| Column Name | Type | Scale / Range | Purpose |
| :--- | :--- | :--- | :--- |
| `student_id` | String | e.g. `STU_0001` | Unique anonymized student identifier. |
| `is_dropout` | Binary | `{0, 1}` | Primary ground truth label: `1` = Dropped out / Academically debarred, `0` = Retained. |
| `ground_truth_risk_prob` | Float | `0.00 – 1.00` | Generative underlying latent risk probability for calibration validation. |
