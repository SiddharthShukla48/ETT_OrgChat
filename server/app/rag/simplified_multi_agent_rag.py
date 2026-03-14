import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from crewai import Agent, Crew, LLM, Task
from crewai.tools import BaseTool
from pydantic import Field
from pypdf import PdfReader

from ..core.config import settings

logger = logging.getLogger(__name__)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class ProjectsCSVTool(BaseTool):
    name: str = "projects_csv_tool"
    description: str = "Searches project allocation data (employee, project, role, department) from CSV."
    dataset: List[Dict[str, str]] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        q = query.lower()
        matches = []
        for row in self.dataset:
            row_text = " ".join(str(v) for v in row.values()).lower()
            if any(token in row_text for token in q.split() if len(token) > 2):
                matches.append(row)

        if not matches:
            return "No closely matching rows found in project allocations."

        lines = []
        for row in matches[:8]:
            lines.append(
                (
                    f"Employee {row.get('employee_name', 'N/A')} ({row.get('employee_id', 'N/A')}) "
                    f"works on {row.get('project_name', 'N/A')} as {row.get('role_in_project', 'N/A')} "
                    f"from {row.get('start_date', 'N/A')} in {row.get('department', 'N/A')}."
                )
            )
        return "\n".join(lines)


class PoliciesCSVTool(BaseTool):
    name: str = "policies_csv_tool"
    description: str = "Searches policy/procedure rules from CSV and returns key policy details."
    dataset: List[Dict[str, str]] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        q = query.lower()
        matches = []
        for row in self.dataset:
            row_text = " ".join(str(v) for v in row.values()).lower()
            if any(token in row_text for token in q.split() if len(token) > 2):
                matches.append(row)

        if not matches:
            return "No relevant policy rows were found."

        lines = []
        for row in matches[:6]:
            lines.append(
                (
                    f"{row.get('title', 'Untitled Policy')} [{row.get('category', 'General')}]: "
                    f"{row.get('description', 'No description')} "
                    f"Applicable roles: {row.get('applicable_roles', 'N/A')}. "
                    f"Approval required: {row.get('approval_required', 'N/A')}."
                )
            )
        return "\n".join(lines)


class OrganizationJSONTool(BaseTool):
    name: str = "organization_json_tool"
    description: str = "Searches organization context JSON (employees, departments, benefits, trainings)."
    data: Dict = Field(default_factory=dict)

    @staticmethod
    def _compact(value):
        if isinstance(value, list):
            return value[:5]
        if isinstance(value, dict):
            return {k: value[k] for k in list(value.keys())[:8]}
        return value

    def _run(self, query: str) -> str:
        q = query.lower()

        if not self.data:
            return "Organizational JSON context is unavailable."

        selected_parts = []
        for key in [
            "organization_info",
            "departments",
            "benefits",
            "training_programs",
            "office_locations",
            "employees",
            "policies",
        ]:
            value = self.data.get(key)
            if value is None:
                continue

            compact_value = self._compact(value)
            serialized = json.dumps(compact_value, ensure_ascii=True)
            if any(token in serialized.lower() for token in q.split() if len(token) > 2):
                selected_parts.append({key: compact_value})

        if not selected_parts:
            # Fallback to compact organization overview when no direct key match is found.
            overview = {
                "organization_info": self._compact(self.data.get("organization_info", {})),
                "departments": self._compact(self.data.get("departments", [])),
                "benefits": self._compact(self.data.get("benefits", [])),
            }
            return json.dumps(overview, ensure_ascii=True, indent=2)

        return json.dumps(selected_parts[:3], ensure_ascii=True, indent=2)


class PolicyPDFTool(BaseTool):
    name: str = "policy_pdf_tool"
    description: str = "Searches policy handbook PDF text passages relevant to user questions."
    content: str = ""

    def _run(self, query: str) -> str:
        if not self.content:
            return "Policy handbook PDF content is unavailable."

        q_tokens = [t for t in query.lower().split() if len(t) > 2]
        paragraphs = [p.strip() for p in self.content.split("\n\n") if p.strip()]

        scored = []
        for para in paragraphs:
            text = para.lower()
            score = sum(1 for token in q_tokens if token in text)
            if score > 0:
                scored.append((score, para))

        if not scored:
            return "No matching handbook section found for this query."

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item[1] for item in scored[:3]]
        return "\n\n".join(top)


class SimplifiedMultiAgentRAGSystem:
    """CrewAI-powered multi-agent RAG system over local org context files."""

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[2]
        self.context_dir = self.base_dir / "RAG_context"
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

        self.projects_data = self._load_csv(self.context_dir / "projects.csv")
        self.policies_data = self._load_csv(self.context_dir / "policies.csv")
        self.org_data = self._load_json(self.context_dir / "rag_context_organizational_data.json")
        self.pdf_text = self._load_pdf_text()

        self.projects_tool = ProjectsCSVTool(dataset=self.projects_data)
        self.policies_tool = PoliciesCSVTool(dataset=self.policies_data)
        self.org_tool = OrganizationJSONTool(data=self.org_data)
        self.pdf_tool = PolicyPDFTool(content=self.pdf_text)

        self.llm = self._build_llm()

    def _build_llm(self) -> Optional[LLM]:
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY not configured; using deterministic fallback responses.")
            return None

        model_name = settings.groq_model or "llama-3.3-70b-versatile"
        try:
            # CrewAI 1.x uses LiteLLM model naming for Groq providers.
            return LLM(
                model=f"groq/{model_name}",
                api_key=settings.groq_api_key,
                temperature=0.2,
                max_tokens=700,
            )
        except Exception as exc:
            logger.error("Failed to initialize CrewAI LLM: %s", exc)
            return None

    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            logger.warning("CSV context missing: %s", path)
            return []

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    @staticmethod
    def _load_json(path: Path) -> Dict:
        if not path.exists():
            logger.warning("JSON context missing: %s", path)
            return {}

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pdf_text(self) -> str:
        pdf_candidates = [
            self.context_dir / "sample_policy_and_procedures_manual.pdf",
            self.context_dir / "sample_policy_and_procedures_manual (1).pdf",
        ]

        for pdf_path in pdf_candidates:
            if not pdf_path.exists():
                continue
            try:
                reader = PdfReader(str(pdf_path))
                chunks = []
                for page in reader.pages:
                    chunks.append(page.extract_text() or "")
                return "\n\n".join(chunks).strip()
            except Exception as exc:
                logger.warning("Unable to parse PDF %s: %s", pdf_path.name, exc)

        return ""

    def analyze_query_type(self, query: str) -> Dict[str, bool]:
        q = query.lower()

        project_keywords = {
            "project",
            "employee",
            "allocation",
            "role",
            "team",
            "department",
            "resource",
            "assignment",
        }
        policy_keywords = {
            "policy",
            "leave",
            "vacation",
            "remote",
            "approval",
            "compliance",
            "procedure",
            "rule",
            "disciplinary",
        }
        org_keywords = {
            "organization",
            "benefit",
            "training",
            "location",
            "company",
            "office",
            "hr",
            "salary",
            "compensation",
            "pay",
            "payroll",
            "wage",
        }

        requires_projects = any(k in q for k in project_keywords)
        requires_policy = any(k in q for k in policy_keywords)
        requires_org = any(k in q for k in org_keywords)

        if not (requires_projects or requires_policy or requires_org):
            # For generic questions, consult all specialists.
            requires_projects = True
            requires_policy = True
            requires_org = True

        return {
            "requires_projects": requires_projects,
            "requires_policy": requires_policy,
            "requires_org": requires_org,
        }

    @staticmethod
    def _is_salary_query(query: str) -> bool:
        q = query.lower()
        salary_terms = {"salary", "salaries", "compensation", "pay", "payroll", "wage", "ctc"}
        return any(term in q for term in salary_terms)

    @staticmethod
    def _extract_name_tokens(query: str) -> List[str]:
        # Extract alphabetic tokens to match likely employee names.
        return [t.lower() for t in re.findall(r"[A-Za-z]+", query) if len(t) > 2]

    def _format_salary(self, amount) -> str:
        try:
            return f"${int(amount):,}"
        except Exception:
            return str(amount)

    def _salary_response(self, query: str) -> str:
        employees = self.org_data.get("employees", []) if isinstance(self.org_data, dict) else []
        if not employees:
            return "Salary details are not available in the current organizational dataset."

        tokens = self._extract_name_tokens(query)
        generic_tokens = {
            "need",
            "salary",
            "salaries",
            "details",
            "detail",
            "employee",
            "employees",
            "show",
            "list",
            "give",
            "tell",
            "all",
            "of",
            "the",
            "for",
            "and",
            "with",
            "their",
        }
        filter_tokens = [t for t in tokens if t not in generic_tokens]

        matches = []
        for emp in employees:
            salary = emp.get("salary")
            if salary is None:
                continue

            full_name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip().lower()
            emp_id = str(emp.get("employee_id", "")).lower()

            if filter_tokens:
                if not any(tok in full_name or tok in emp_id for tok in filter_tokens):
                    continue

            matches.append(emp)

        if filter_tokens and not matches:
            return "I could not find salary records for the specified employee name or ID."

        selected = matches if matches else [e for e in employees if e.get("salary") is not None]
        lines = []
        for emp in selected[:15]:
            name = f"{emp.get('first_name', '')} {emp.get('last_name', '')}".strip() or emp.get("employee_id", "Unknown")
            lines.append(
                f"- {name} ({emp.get('employee_id', 'N/A')}): {self._format_salary(emp.get('salary'))}"
            )

        if len(selected) > 15:
            lines.append(f"- ...and {len(selected) - 15} more employees in the dataset.")

        salaries = [int(e.get("salary")) for e in selected if e.get("salary") is not None]
        if salaries:
            summary = (
                f"Salary range: {self._format_salary(min(salaries))} to {self._format_salary(max(salaries))}. "
                f"Average: {self._format_salary(sum(salaries) // len(salaries))}."
            )
        else:
            summary = "No numeric salary values were found."

        return "Employee salary details:\n" + "\n".join(lines) + "\n\n" + summary

    def _build_agents(self, analysis: Dict[str, bool]) -> List[Agent]:
        agents: List[Agent] = []

        if analysis.get("requires_projects"):
            agents.append(
                Agent(
                    role="Projects and Employee Data Specialist",
                    goal="Extract exact project allocation and employee-role details from structured data.",
                    backstory="Expert in project staffing, employee assignments, and departmental workloads.",
                    tools=[self.projects_tool],
                    llm=self.llm,
                    allow_delegation=False,
                    verbose=False,
                )
            )

        if analysis.get("requires_policy"):
            agents.append(
                Agent(
                    role="Policy and Procedures Specialist",
                    goal="Retrieve policy clauses and approval rules from policy datasets and handbook.",
                    backstory="Expert in workplace policy interpretation and HR compliance language.",
                    tools=[self.policies_tool, self.pdf_tool],
                    llm=self.llm,
                    allow_delegation=False,
                    verbose=False,
                )
            )

        if analysis.get("requires_org"):
            agents.append(
                Agent(
                    role="Organizational Data Analyst",
                    goal="Explain organization-level information such as benefits, departments, and trainings.",
                    backstory="Expert in organizational structure and people operations data.",
                    tools=[self.org_tool],
                    llm=self.llm,
                    allow_delegation=False,
                    verbose=False,
                )
            )

        agents.append(
            Agent(
                role="Knowledge Synthesis Manager",
                goal="Synthesize specialist outputs into one concise, actionable response.",
                backstory="Experienced manager who consolidates insights from multiple experts.",
                llm=self.llm,
                allow_delegation=False,
                verbose=False,
            )
        )

        return agents

    def _build_tasks(self, query: str, agents: List[Agent]) -> List[Task]:
        tasks: List[Task] = []

        specialists = [a for a in agents if a.role != "Knowledge Synthesis Manager"]
        manager = next(a for a in agents if a.role == "Knowledge Synthesis Manager")

        for specialist in specialists:
            tasks.append(
                Task(
                    description=(
                        f"User query: '{query}'. Use your assigned data tools to gather relevant facts. "
                        "Return bullet points with explicit references to entities (employee/project/policy/etc.)."
                    ),
                    expected_output="Bulleted factual findings from your data source.",
                    agent=specialist,
                )
            )

        tasks.append(
            Task(
                description=(
                    f"Synthesize all specialist findings for query '{query}'. "
                    "Provide one final answer that is clear, concise, and specific. "
                    "If data is missing, state the limitation explicitly."
                ),
                expected_output="A final user-facing response.",
                agent=manager,
            )
        )

        return tasks

    def _fallback_response(self, query: str, analysis: Dict[str, bool]) -> str:
        sections = []

        if analysis.get("requires_projects"):
            projects_text = _truncate(self.projects_tool._run(query), 900)
            sections.append("[Projects Data]\n" + projects_text)
        if analysis.get("requires_policy"):
            policy_snippet = _truncate(self.policies_tool._run(query), 900)
            pdf_snippet = _truncate(self.pdf_tool._run(query), 900)
            sections.append("[Policy Data]\n" + policy_snippet + "\n\n[Policy Handbook]\n" + pdf_snippet)
        if analysis.get("requires_org"):
            org_text = _truncate(self.org_tool._run(query), 900)
            sections.append("[Organization Data]\n" + org_text)

        if not sections:
            sections.append("No matching context sources were selected for the query.")

        return _truncate("\n\n".join(sections), 2500)

    def chat(self, query: str, session_id: str) -> str:
        if not session_id:
            session_id = "default"

        history = self.conversation_history.setdefault(session_id, [])
        history.append({"role": "user", "content": query})

        analysis = self.analyze_query_type(query)

        if self._is_salary_query(query):
            final_response = self._salary_response(query)
            history.append({"role": "assistant", "content": final_response})
            return final_response

        try:
            if self.llm is None:
                final_response = self._fallback_response(query, analysis)
            else:
                agents = self._build_agents(analysis)
                tasks = self._build_tasks(query, agents)
                crew = Crew(agents=agents, tasks=tasks, verbose=False)
                result = crew.kickoff()
                final_response = str(result).strip()
                if not final_response:
                    final_response = self._fallback_response(query, analysis)
        except Exception as exc:
            logger.error("Crew execution failed: %s", exc)
            final_response = self._fallback_response(query, analysis)

        history.append({"role": "assistant", "content": final_response})
        return final_response

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.conversation_history.get(session_id, [])

    def clear_conversation(self, session_id: str) -> None:
        self.conversation_history.pop(session_id, None)
