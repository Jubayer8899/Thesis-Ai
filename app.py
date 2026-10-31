import os
import requests
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from collections import defaultdict

# ---------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------
load_dotenv()

# Streamlit secrets or .env configurations
DEFAULT_GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()

# ⚠️ Updated model to gemini-3.8-flash as required by API
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
OPENALEX_API_KEY = st.secrets.get("OPENALEX_API_KEY", os.getenv("OPENALEX_API_KEY", "")).strip()

# Password Configuration
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "coder123"))          # General Users
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "myadmin789"))  # Admin User (You)

OPENALEX_BASE_URL = "https://api.openalex.org"
REQUEST_TIMEOUT = 30

st.set_page_config(
    page_title="Thesis AI Agent",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------
# 🔒 TWO-TIER AUTHENTICATION (ADMIN vs GENERAL USER)
# ---------------------------------------------------------
def check_password():
    """Manages login authentication for Admin and General users."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.is_admin = False

    if not st.session_state.authenticated:
        st.title("🔒 Thesis AI Agent - Login Required")
        user_input = st.text_input("Enter Passcode to Access:", type="password")
        if st.button("Login"):
            if user_input == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.is_admin = True   # Logged in as Admin
                st.rerun()
            elif user_input == APP_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.is_admin = False  # Logged in as General User
                st.rerun()
            else:
                st.error("Incorrect Passcode! Access Denied.")
        return False
    return True

if not check_password():
    st.stop()


# ---------------------------------------------------------
# 🔑 DYNAMIC API KEY & SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Configuration")
    
    if st.session_state.is_admin:
        ACTIVE_GEMINI_KEY = DEFAULT_GEMINI_KEY
        st.success("👑 Admin Mode Active: Secret API Key Auto-Loaded!")
        if not ACTIVE_GEMINI_KEY:
            st.warning("⚠️ Admin Warning: Default GEMINI_API_KEY is not set in secrets/env.")
    else:
        user_api_key_input = st.text_input(
            "Enter your Gemini API Key", 
            type="password",
            placeholder="AIzaSy...",
            help="Apnar personal Gemini API key use korle apnar billing/quota burn hobe."
        )
        
        if user_api_key_input.strip():
            ACTIVE_GEMINI_KEY = user_api_key_input.strip()
            st.success("🟢 Custom API key connected!")
        else:
            ACTIVE_GEMINI_KEY = None
            st.warning("⚠️ Please enter your Gemini API key to unlock features.")

if not ACTIVE_GEMINI_KEY:
    st.warning("🔒 Access Blocked: Gemini API Key is required to use this application.")
    st.stop()


# ---------------------------------------------------------
# Gemini AI Engine (Uses ACTIVE_GEMINI_KEY & gemini-3.8-flash)
# ---------------------------------------------------------
def get_gemini_client():
    if not ACTIVE_GEMINI_KEY:
        return None
    return genai.Client(api_key=ACTIVE_GEMINI_KEY)


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return text response."""
    client = get_gemini_client()

    if client is None:
        return "Gemini API key is missing. Please provide a valid API key in the sidebar."

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text or "Gemini returned an empty response."
    except Exception as error:
        return f"Gemini API error: {error}"


# ---------------------------------------------------------
# OpenAlex Academic Database API & Helpers
# ---------------------------------------------------------
def openalex_get(endpoint: str, params: dict) -> dict:
    request_params = dict(params)
    if OPENALEX_API_KEY:
        request_params["api_key"] = OPENALEX_API_KEY

    response = requests.get(
        f"{OPENALEX_BASE_URL}/{endpoint}",
        params=request_params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 429:
        raise RuntimeError("OpenAlex rate/budget limit reached. Please wait or check usage.")

    response.raise_for_status()
    return response.json()


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return "Abstract not available in OpenAlex."

    words = []
    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])
    return " ".join(word for _, word in words)


def get_author_names(work):
    names = []
    for authorship in work.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    return ", ".join(names) if names else "Authors not listed"


@st.cache_data(ttl=3600, show_spinner=False)
def search_papers(topic: str, year_from: int, year_to: int, limit: int):
    params = {
        "search": topic,
        "per_page": limit,
        "sort": "relevance_score:desc"
    }

    if year_from and year_to:
        params["filter"] = f"publication_year:{year_from}-{year_to}"

    data = openalex_get("works", params)
    return data.get("results", []), data.get("meta", {}).get("count", 0)


@st.cache_data(ttl=3600, show_spinner=False)
def find_researchers(topic: str, year_from: int, year_to: int, limit: int, country_code: str = ""):
    params = {
        "search": topic,
        "per_page": limit,
        "sort": "relevance_score:desc"
    }

    filters = []
    if year_from and year_to:
        filters.append(f"publication_year:{year_from}-{year_to}")
    if country_code and country_code.strip():
        filters.append(f"institutions.country_code:{country_code.strip().upper()}")

    if filters:
        params["filter"] = ",".join(filters)

    data = openalex_get("works", params)
    works = data.get("results", [])

    researcher_map = {}

    for work in works:
        work_title = work.get("title") or "Untitled work"
        work_year = work.get("publication_year")
        work_doi = work.get("doi")
        work_url = work_doi or work.get("id") or ""

        for authorship in work.get("authorships", []) or []:
            author = authorship.get("author") or {}
            author_id = author.get("id")
            author_name = author.get("display_name")

            if not author_id or not author_name:
                continue

            institutions = authorship.get("institutions", []) or []
            
            if country_code and country_code.strip():
                match_country = any(
                    inst.get("country_code", "").upper() == country_code.strip().upper()
                    for inst in institutions
                )
                if not match_country:
                    continue

            if author_id not in researcher_map:
                researcher_map[author_id] = {
                    "name": author_name,
                    "id": author_id,
                    "works_found": 0,
                    "citations_in_results": 0,
                    "institutions": set(),
                    "countries": set(),
                    "sample_papers": [],
                    "orcid": author.get("orcid")
                }

            researcher = researcher_map[author_id]
            researcher["works_found"] += 1
            researcher["citations_in_results"] += work.get("cited_by_count") or 0

            for institution in institutions:
                inst_name = institution.get("display_name")
                country = institution.get("country_code")
                if inst_name:
                    researcher["institutions"].add(inst_name)
                if country:
                    researcher["countries"].add(country)

            if len(researcher["sample_papers"]) < 5:
                researcher["sample_papers"].append({
                    "title": work_title,
                    "year": work_year,
                    "url": work_url
                })

    researchers = list(researcher_map.values())
    researchers.sort(
        key=lambda item: (item["works_found"], item["citations_in_results"]),
        reverse=True
    )

    for researcher in researchers:
        researcher["institutions"] = ", ".join(sorted(researcher["institutions"])) or "Affiliation not listed"
        researcher["countries"] = ", ".join(sorted(researcher["countries"])) or "Not listed"

    return researchers, data.get("meta", {}).get("count", 0)


# ---------------------------------------------------------
# PDF Processing & Export Helpers
# ---------------------------------------------------------
def extract_pdf_text(uploaded_file):
    try:
        pdf_document = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        page_texts = []
        for page_number, page in enumerate(pdf_document, start=1):
            text = page.get_text("text")
            if text.strip():
                page_texts.append(f"\n[PAGE {page_number}]\n{text}")

        pdf_document.close()
        extracted_text = "\n".join(page_texts).strip()
        if not extracted_text:
            return "No selectable text was found. This may be a scanned PDF; OCR will be needed."
        return extracted_text
    except Exception as error:
        return f"Could not read this PDF: {error}"


def build_paper_prompt(task: str, paper_text: str) -> str:
    return f"""
You are an academic research assistant specializing in Civil Engineering.

IMPORTANT RULES:
- Use only evidence available in the supplied paper when describing its findings.
- Do not invent numerical results, citations, DOI numbers, or page references.
- Clearly say when information is not available in the paper.
- Separate the authors' claims from your own interpretation.
- Use clear academic English.

TASK:
{task}

PAPER TEXT:
{paper_text[:45000]}
"""


def convert_df_to_excel(df):
    """Encodes DataFrame into Excel Bytes for Streamlit Download Button."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Export')
    return output.getvalue()


# ---------------------------------------------------------
# Main App & Sidebar Info
# ---------------------------------------------------------
st.title("📚 Thesis AI Agent")
st.caption("Gemini-powered research assistant for Civil Engineering, academic papers, & researcher discovery.")

with st.sidebar:
    st.divider()
    st.header("⚙️ System Status")
    st.write(f"**Gemini Model:** `{GEMINI_MODEL}`")
    st.write(
        "**Gemini Status:** " +
        ("🟢 Active" if ACTIVE_GEMINI_KEY else "🔴 Key Missing")
    )
    st.write(
        "**OpenAlex API:** " +
        ("🟢 Connected" if OPENALEX_API_KEY else "⚪ Public Mode")
    )
    st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Paper Reader & Summary",
    "🔎 Search & Export Papers",
    "👨‍🏫 Professor Finder",
    "🤖 Gemini Assistant",
    "📝 Application Generator"
])


# ---------------------------------------------------------
# TAB 1: Paper Reader & Gap Finder (With Download Option)
# ---------------------------------------------------------
with tab1:
    st.header("Research Paper Reader & Gap Analyzer")
    uploaded_pdf = st.file_uploader("Upload a research paper (PDF)", type=["pdf"], key="paper_pdf")

    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            paper_text = extract_pdf_text(uploaded_pdf)

        if paper_text.startswith("Could not read") or paper_text.startswith("No selectable"):
            st.warning(paper_text)
        else:
            st.success("PDF text extracted successfully.")
            with st.expander("Preview Extracted Text"):
                st.text(paper_text[:12000])

            analysis_task = st.selectbox(
                "Choose an analysis task",
                [
                    "Complete research paper summary",
                    "Research problem, aim, and objectives",
                    "Detailed methodology",
                    "Materials, experimental program, and variables",
                    "Main findings and numerical results",
                    "Limitations and future research",
                    "Research gap and possible follow-up study",
                    "Explain this paper in simple language"
                ]
            )

            if st.button("Analyze Paper with Gemini", key="analyze_pdf"):
                prompt = build_paper_prompt(analysis_task, paper_text)
                with st.spinner("Gemini is analyzing the paper..."):
                    st.session_state.paper_summary_result = ask_gemini(prompt)

            # Show Result & Add Download Option
            if "paper_summary_result" in st.session_state:
                st.markdown("### 📋 Analysis Result")
                st.markdown(st.session_state.paper_summary_result)
                
                # 📥 Download Summary Button
                st.download_button(
                    label="📥 Download Research Summary (.txt)",
                    data=st.session_state.paper_summary_result,
                    file_name="paper_summary.txt",
                    mime="text/plain"
                )

            st.divider()
            st.subheader("Ask a question about this paper")
            question = st.text_input("Your question", placeholder="Example: What was the water-cement ratio?")

            if st.button("Ask About This PDF", key="ask_pdf"):
                if not question.strip():
                    st.warning("Please write a question first.")
                else:
                    prompt = build_paper_prompt(
                        f"Answer this question using the paper. Question: {question}",
                        paper_text
                    )
                    with st.spinner("Finding answer..."):
                        ans = ask_gemini(prompt)
                        st.markdown(ans)
                        st.download_button("📥 Download Answer (.txt)", data=ans, file_name="pdf_answer.txt")


# ---------------------------------------------------------
# TAB 2: Topic-wise Paper Search & Excel Export
# ---------------------------------------------------------
with tab2:
    st.header("Topic-wise Academic Paper Search & Data Export")
    paper_topic = st.text_input("Research topic", placeholder="Example: Machine learning concrete strength", key="paper_topic")

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        paper_year_from = st.number_input("From year", min_value=1900, max_value=2100, value=2018, key="paper_from")
    with pcol2:
        paper_year_to = st.number_input("To year", min_value=1900, max_value=2100, value=2026, key="paper_to")
    with pcol3:
        paper_limit = st.selectbox("Maximum results", [10, 25, 50, 100], index=1, key="paper_limit")

    if st.button("Search Research Papers", type="primary", key="search_papers"):
        if not paper_topic.strip():
            st.warning("Enter a research topic.")
        else:
            try:
                with st.spinner("Searching OpenAlex database..."):
                    papers, total_count = search_papers(
                        paper_topic.strip(), int(paper_year_from), int(paper_year_to), int(paper_limit)
                    )

                st.success(f"Found {len(papers)} results. Total records: {total_count:,}")

                if papers:
                    excel_rows = []
                    for index, paper in enumerate(papers, start=1):
                        title = paper.get("title") or "Title unavailable"
                        year = paper.get("publication_year") or "N/A"
                        doi = paper.get("doi") or "N/A"
                        cited = paper.get("cited_by_count", 0)
                        authors = get_author_names(paper)
                        abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
                        paper_url = doi if doi != "N/A" else paper.get("id")

                        excel_rows.append({
                            "Title": title,
                            "Year": year,
                            "Authors": authors,
                            "Citations": cited,
                            "DOI": doi,
                            "URL": paper_url,
                            "Abstract": abstract[:1000]
                        })

                        with st.expander(f"{index}. {title} ({year})"):
                            st.write(f"**Authors:** {authors}")
                            st.write(f"**Citations:** {cited}")
                            st.write(f"**Abstract:** {abstract[:1500]}")

                    # Generate Excel Download
                    df_papers = pd.DataFrame(excel_rows)
                    excel_data = convert_df_to_excel(df_papers)
                    
                    st.download_button(
                        label="📥 Download Papers List as Excel (.xlsx)",
                        data=excel_data,
                        file_name=f"papers_{paper_topic[:15].strip()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as error:
                st.error(f"Paper search failed: {error}")


# ---------------------------------------------------------
# TAB 3: Country-wise Professor / Researcher Excel Export
# ---------------------------------------------------------
with tab3:
    st.header("Find Professors and Researchers by Topic & Country")
    professor_topic = st.text_input("Research topic", placeholder="e.g., Physics-informed neural networks", key="prof_topic")

    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        researcher_year_from = st.number_input("Publication start year", min_value=1900, max_value=2100, value=2018, key="res_from")
    with ccol2:
        researcher_year_to = st.number_input("Publication end year", min_value=1900, max_value=2100, value=2026, key="res_to")
    with ccol3:
        country_filter = st.text_input("Country Code (Optional)", placeholder="e.g., US, CA, DE, BD", key="country_filter")

    researcher_limit = st.selectbox("Inspect papers limit", [25, 50, 100], index=1, key="res_limit")

    if st.button("Find Professors / Researchers", type="primary", key="find_professors"):
        if not professor_topic.strip():
            st.warning("Enter a research topic.")
        else:
            try:
                with st.spinner("Searching matching researchers..."):
                    researchers, total_works = find_researchers(
                        professor_topic.strip(), int(researcher_year_from), int(researcher_year_to), int(researcher_limit), country_filter.strip()
                    )

                st.success(f"Identified {len(researchers)} researchers.")

                if researchers:
                    excel_res_rows = []
                    for index, researcher in enumerate(researchers[:30], start=1):
                        excel_res_rows.append({
                            "Name": researcher["name"],
                            "Institutions": researcher["institutions"],
                            "Countries": researcher["countries"],
                            "Matching Papers Count": researcher["works_found"],
                            "Total Citations": researcher["citations_in_results"],
                            "Profile URL": researcher["id"],
                            "ORCID": researcher.get("orcid", "N/A")
                        })

                        with st.expander(f"{index}. {researcher['name']} — {researcher['institutions']} ({researcher['countries']})"):
                            st.write(f"**Matching Papers:** {researcher['works_found']}")
                            st.write(f"**Total Citations:** {researcher['citations_in_results']}")
                            st.markdown(f"[Open Profile]({researcher['id']})")

                    # Generate Professor Excel Download
                    df_researchers = pd.DataFrame(excel_res_rows)
                    excel_res_data = convert_df_to_excel(df_researchers)
                    
                    st.download_button(
                        label="📥 Download Professor List as Excel (.xlsx)",
                        data=excel_res_data,
                        file_name=f"professors_{professor_topic[:15].strip()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as error:
                st.error(f"Researcher search failed: {error}")


# ---------------------------------------------------------
# TAB 4: General Gemini Research Assistant
# ---------------------------------------------------------
with tab4:
    st.header("Gemini Research Assistant")
    assistant_question = st.text_area("Ask your research question", height=150, key="ast_q")

    if st.button("Ask Gemini", type="primary", key="general_gemini"):
        if assistant_question.strip():
            with st.spinner("Gemini is preparing your answer..."):
                res = ask_gemini(assistant_question)
                st.markdown(res)
                st.download_button("📥 Download Answer (.txt)", data=res, file_name="ai_response.txt")


# ---------------------------------------------------------
# TAB 5: Application & Proposal Generator
# ---------------------------------------------------------
with tab5:
    st.header("Academic Document Generator")
    doc_type = st.selectbox("Select Document Type", ["Cover Letter / Email to Professor", "Research Proposal Outline"])
    details = st.text_area("Provide background/topic details:")

    if st.button("Generate Document", type="primary", key="gen_doc"):
        if details.strip():
            with st.spinner("Generating document..."):
                doc_res = ask_gemini(f"Draft a {doc_type} based on: {details}")
                st.markdown(doc_res)
                st.download_button("📥 Download Document (.txt)", data=doc_res, file_name="generated_document.txt")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption("Thesis AI Agent | Powered by Streamlit & Gemini 3.8 Flash")
