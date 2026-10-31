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
# Configuration
# ---------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()

OPENALEX_BASE_URL = "https://api.openalex.org"
REQUEST_TIMEOUT = 30

st.set_page_config(
    page_title="Thesis AI Agent",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# Gemini AI
# ---------------------------------------------------------
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return its text response."""
    client = get_gemini_client()

    if client is None:
        return (
            "Gemini API key is missing. Add GEMINI_API_KEY to your .env file, "
            "then restart the Streamlit app."
        )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text or "Gemini returned an empty response."
    except Exception as error:
        return f"Gemini API error: {error}"


# ---------------------------------------------------------
# OpenAlex academic database
# ---------------------------------------------------------
def openalex_get(endpoint: str, params: dict) -> dict:
    """Request data from OpenAlex. An API key is required for normal use."""
    if not OPENALEX_API_KEY:
        raise ValueError(
            "OpenAlex API key is missing. Add OPENALEX_API_KEY to your .env file."
        )

    request_params = dict(params)
    request_params["api_key"] = OPENALEX_API_KEY

    response = requests.get(
        f"{OPENALEX_BASE_URL}/{endpoint}",
        params=request_params,
        timeout=REQUEST_TIMEOUT
    )

    if response.status_code == 429:
        raise RuntimeError(
            "OpenAlex rate/budget limit reached. Please wait or check your API usage."
        )

    response.raise_for_status()
    return response.json()


def reconstruct_abstract(inverted_index):
    """Convert OpenAlex abstract inverted index into readable text."""
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
    """
    Find researchers by searching relevant papers, then aggregate the authors
    appearing on those papers, with optional country filter.
    """
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
            
            # If country filter is applied, check if any institution matches the country code
            if country_code and country_code.strip():
                match_country = False
                for inst in institutions:
                    if inst.get("country_code", "").upper() == country_code.strip().upper():
                        match_country = True
                        break
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
                institution_name = institution.get("display_name")
                country = institution.get("country_code")
                if institution_name:
                    researcher["institutions"].add(institution_name)
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
        key=lambda item: (
            item["works_found"],
            item["citations_in_results"]
        ),
        reverse=True
    )

    for researcher in researchers:
        researcher["institutions"] = ", ".join(
            sorted(researcher["institutions"])
        ) or "Affiliation not listed"
        researcher["countries"] = ", ".join(
            sorted(researcher["countries"])
        ) or "Not listed"

    return researchers, data.get("meta", {}).get("count", 0)


# ---------------------------------------------------------
# PDF processing
# ---------------------------------------------------------
def extract_pdf_text(uploaded_file):
    """Extract text from a PDF uploaded in Streamlit."""
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
            return (
                "No selectable text was found. This may be a scanned PDF; "
                "OCR will be needed."
            )

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


# Helper for Excel export
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# ---------------------------------------------------------
# Application interface
# ---------------------------------------------------------
st.title("📚 Thesis AI Agent")
st.caption(
    "Gemini-powered research assistant for Civil Engineering, "
    "academic papers, literature review, researcher discovery, and application docs."
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**Gemini model:** `{GEMINI_MODEL}`")
    st.write(
        "**Gemini API:** " +
        ("🟢 API key detected" if GEMINI_API_KEY else "🔴 API key missing")
    )
    st.write(
        "**OpenAlex API:** " +
        ("🟢 API key detected" if OPENALEX_API_KEY else "🔴 API key missing")
    )
    st.divider()
    st.markdown("**Research workflow**")
    st.markdown(
        "1. Search your research topic.\n"
        "2. Review papers & find gaps.\n"
        "3. Find professors country-wise.\n"
        "4. Read and analyze PDFs.\n"
        "5. Generate Cover Letters & Proposals."
    )
    st.caption(
        "Always verify paper details, author roles, affiliations, "
        "and research claims from the original sources."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Paper Reader & Gap Finder",
    "🔎 Paper Search & Export",
    "👨‍🏫 Professor Finder",
    "🤖 Gemini Assistant",
    "📝 Application & Proposal Gen"
])


# ---------------------------------------------------------
# TAB 1: Paper Reader & Gap Finder
# ---------------------------------------------------------
with tab1:
    st.header("Research Paper Reader & Gap Analyzer")
    uploaded_pdf = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        key="paper_pdf"
    )

    if uploaded_pdf:
        with st.spinner("Extracting PDF text..."):
            paper_text = extract_pdf_text(uploaded_pdf)

        if paper_text.startswith("Could not read") or paper_text.startswith("No selectable"):
            st.warning(paper_text)
        else:
            st.success("PDF text extracted successfully.")
            with st.expander("Preview extracted text"):
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
                    result = ask_gemini(prompt)
                st.markdown(result)

            st.divider()
            st.subheader("Ask a question about this paper")
            question = st.text_input(
                "Your question",
                placeholder="Example: What was the water-cement ratio?"
            )

            if st.button("Ask About This PDF", key="ask_pdf"):
                if not question.strip():
                    st.warning("Please write a question first.")
                else:
                    prompt = build_paper_prompt(
                        f"Answer this question using the paper. "
                        f"Where possible, identify the relevant page number. "
                        f"Question: {question}",
                        paper_text
                    )
                    with st.spinner("Finding the answer..."):
                        answer = ask_gemini(prompt)
                    st.markdown(answer)


# ---------------------------------------------------------
# TAB 2: Topic-wise Paper Search & Export
# ---------------------------------------------------------
with tab2:
    st.header("Topic-wise Academic Paper Search & Data Export")
    st.write(
        "Search research publications by topic using OpenAlex and export to Excel."
    )

    paper_topic = st.text_input(
        "Research topic",
        placeholder=(
            "Example: Machine learning prediction of concrete compressive strength"
        ),
        key="paper_topic"
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        paper_year_from = st.number_input(
            "From year", min_value=1900, max_value=2100,
            value=2018, step=1, key="paper_from"
        )
    with pcol2:
        paper_year_to = st.number_input(
            "To year", min_value=1900, max_value=2100,
            value=2026, step=1, key="paper_to"
        )
    with pcol3:
        paper_limit = st.selectbox(
            "Maximum results", [10, 25, 50, 100], index=1,
            key="paper_limit"
        )

    if st.button("Search Research Papers", type="primary", key="search_papers"):
        if not paper_topic.strip():
            st.warning("Enter a research topic.")
        elif paper_year_from > paper_year_to:
            st.warning("The starting year must not be later than the ending year.")
        else:
            try:
                with st.spinner("Searching OpenAlex..."):
                    papers, total_count = search_papers(
                        paper_topic.strip(),
                        int(paper_year_from),
                        int(paper_year_to),
                        int(paper_limit)
                    )

                st.success(
                    f"Found {len(papers)} results in this search. "
                    f"OpenAlex reports approximately {total_count:,} matching records."
                )

                if not papers:
                    st.info("No papers found. Try broader or different keywords.")
                else:
                    # Prepare dataframe for Excel download
                    excel_rows = []
                    for index, paper in enumerate(papers, start=1):
                        title = paper.get("title") or "Title unavailable"
                        year = paper.get("publication_year") or "Year unavailable"
                        doi = paper.get("doi")
                        cited = paper.get("cited_by_count", 0)
                        authors = get_author_names(paper)
                        abstract = reconstruct_abstract(paper.get("abstract_inverted_index"))
                        landing_page = ((paper.get("primary_location") or {}).get("landing_page_url"))
                        paper_url = doi or landing_page or paper.get("id")

                        excel_rows.append({
                            "Title": title,
                            "Year": year,
                            "Authors": authors,
                            "Citations": cited,
                            "DOI": doi or "N/A",
                            "URL": paper_url,
                            "Abstract": abstract[:1000]
                        })

                        with st.expander(f"{index}. {title} ({year})"):
                            st.write(f"**Authors:** {authors}")
                            st.write(f"**Citations:** {cited}")
                            st.write(f"**DOI:** {doi or 'Not available'}")
                            st.write(f"**Abstract:** {abstract[:2500]}")
                            if paper_url:
                                st.markdown(f"[Open paper / record]({paper_url})")

                    df_papers = pd.DataFrame(excel_rows)
                    excel_data = convert_df_to_excel(df_papers)
                    st.download_button(
                        label="📥 Download Papers as Excel (.xlsx)",
                        data=excel_data,
                        file_name=f"papers_{paper_topic[:20].strip()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as error:
                st.error(f"Paper search failed: {error}")


# ---------------------------------------------------------
# TAB 3: Country-wise Professor / Researcher Finder
# ---------------------------------------------------------
with tab3:
    st.header("Find Professors and Researchers by Topic & Country")
    st.write(
        "Search topic-related papers and filter researchers by country code (e.g., US, CA, DE, BD, AU, JP)."
    )

    professor_topic = st.text_input(
        "Research topic",
        placeholder=(
            "Example: Physics-informed machine learning for concrete structures"
        ),
        key="professor_topic"
    )

    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        researcher_year_from = st.number_input(
            "Publication start year",
            min_value=1900, max_value=2100, value=2018,
            key="researcher_from"
        )
    with ccol2:
        researcher_year_to = st.number_input(
            "Publication end year",
            min_value=1900, max_value=2100, value=2026,
            key="researcher_to"
        )
    with ccol3:
        country_filter = st.text_input(
            "Country Code (Optional)",
            placeholder="e.g., US, CA, DE, BD",
            key="country_filter"
        )

    researcher_limit = st.selectbox(
        "Number of topic-related papers to inspect",
        [25, 50, 100],
        index=1,
        key="researcher_limit"
    )

    if st.button("Find Professors / Researchers", type="primary", key="find_professors"):
        if not professor_topic.strip():
            st.warning("Enter a research topic.")
        elif researcher_year_from > researcher_year_to:
            st.warning("The starting year must not be later than the ending year.")
        else:
            try:
                with st.spinner(
                    "Searching topic-related papers and identifying their authors..."
                ):
                    researchers, total_works = find_researchers(
                        professor_topic.strip(),
                        int(researcher_year_from),
                        int(researcher_year_to),
                        int(researcher_limit),
                        country_filter.strip()
                    )

                st.success(
                    f"Identified {len(researchers)} researchers matching your criteria."
                )

                if not researchers:
                    st.info("No researchers found. Try broadening your keywords or clearing the country filter.")
                else:
                    excel_res_rows = []
                    for index, researcher in enumerate(researchers[:30], start=1):
                        excel_res_rows.append({
                            "Name": researcher["name"],
                            "Institutions": researcher["institutions"],
                            "Countries": researcher["countries"],
                            "Matching Papers": researcher["works_found"],
                            "Citations": researcher["citations_in_results"],
                            "Profile ID": researcher["id"],
                            "ORCID": researcher.get("orcid", "N/A")
                        })

                        with st.expander(
                            f"{index}. {researcher['name']} "
                            f"— {researcher['works_found']} matching paper(s) | {researcher['countries']}"
                        ):
                            st.write(f"**Institution(s):** {researcher['institutions']}")
                            st.write(f"**Country/countries:** {researcher['countries']}")
                            st.write(
                                f"**Matching papers in this search:** "
                                f"{researcher['works_found']}"
                            )
                            st.write(
                                f"**Citations of these retrieved papers:** "
                                f"{researcher['citations_in_results']:,}"
                            )
                            st.markdown(
                                f"[Open researcher profile]({researcher['id']})"
                            )
                            if researcher.get("orcid"):
                                st.markdown(
                                    f"[ORCID profile]({researcher['orcid']})"
                                )

                            st.markdown("**Relevant papers found:**")
                            for sample in researcher["sample_papers"]:
                                paper_label = sample["title"]
                                if sample["year"]:
                                    paper_label += f" ({sample['year']})"
                                if sample["url"]:
                                    st.markdown(f"- [{paper_label}]({sample['url']})")
                                else:
                                    st.markdown(f"- {paper_label}")

                    df_researchers = pd.DataFrame(excel_res_rows)
                    excel_res_data = convert_df_to_excel(df_researchers)
                    st.download_button(
                        label="📥 Download Researchers as Excel (.xlsx)",
                        data=excel_res_data,
                        file_name=f"researchers_{professor_topic[:20].strip()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as error:
                st.error(f"Researcher search failed: {error}")


# ---------------------------------------------------------
# TAB 4: General Gemini Research Assistant
# ---------------------------------------------------------
with tab4:
    st.header("Gemini Research Assistant")

    assistant_topic = st.text_input(
        "Your research area (optional)",
        placeholder="Example: Sustainable concrete using rice husk ash",
        key="ast_topic"
    )

    assistant_question = st.text_area(
        "Ask your research question",
        placeholder=(
            "Example: Suggest research objectives for predicting concrete "
            "compressive strength using machine learning."
        ),
        height=150,
        key="ast_q"
    )

    assistant_style = st.selectbox(
        "Answer style",
        [
            "Detailed academic explanation",
            "Simple explanation for learning",
            "Literature review guidance",
            "Research methodology guidance",
            "Research proposal structure"
        ],
        key="ast_style"
    )

    if st.button("Ask Gemini", type="primary", key="general_gemini"):
        if not assistant_question.strip():
            st.warning("Write a question first.")
        else:
            prompt = f"""
You are an academic research assistant for a Civil Engineering student.

Research area: {assistant_topic or "Civil Engineering"}
Preferred answer style: {assistant_style}

User question:
{assistant_question}

Instructions:
- Be accurate, clear, and academically useful.
- Structure the answer with headings and bullet points where appropriate.
- Do not invent references, DOIs, authors, or experimental data.
- If suggesting papers, explain that the user must verify them in a scholarly database.
- Distinguish established knowledge from suggestions or hypotheses.
"""
            with st.spinner("Gemini is preparing your answer..."):
                answer = ask_gemini(prompt)
            st.markdown(answer)


# ---------------------------------------------------------
# TAB 5: Application & Proposal Generator
# ---------------------------------------------------------
with tab5:
    st.header("Academic Document & Application Generator")
    st.write(
        "Generate professional Cover Letters, Recommendation Letters, Research Interests, and Research Proposals using Gemini."
    )

    doc_type = st.selectbox(
        "Select Document Type",
        [
            "Cover Letter / Email to Professor",
            "Recommendation Letter Draft",
            "Research Interest Statement / SOP",
            "Research Proposal Outline"
        ]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        applicant_name = st.text_input("Your Full Name", value="Md. Jubayer Hossain Mahin")
        applicant_field = st.text_input("Your Field / Major", value="Civil Engineering")
    with col_b:
        target_prof_uni = st.text_input("Target Professor / University Name", placeholder="e.g. Prof. John Doe, University of Waterloo")
        specific_topic = st.text_input("Specific Research Topic / Focus", placeholder="e.g. Sustainable Concrete & ML")

    user_background = st.text_area(
        "Provide your background, achievements, or project details to include",
        placeholder="Mention your CGPA, software skills (ETABS, Python), thesis topic, or key achievements..."
    )

    if st.button("Generate Document with Gemini", type="primary", key="gen_doc"):
        if not target_prof_uni.strip() and "Cover Letter" in doc_type:
            st.warning("Please enter the target Professor/University name.")
        else:
            prompt = f"""
You are an expert academic mentor and writer. Please write a highly professional, polite, and persuasive academic document based on the following details:

Document Type: {doc_type}
Applicant Name: {applicant_name}
Applicant Field: {applicant_field}
Target Professor / University: {target_prof_uni}
Research Focus / Topic: {specific_topic}
Applicant Background & Details: {user_background}

Instructions:
- Use formal academic tone.
- Make it structured, clean, and ready to use.
- Use placeholders like [Date], [University Name] if any vital detail is missing.
- Do not fabricate false publication records, but format placeholders properly for the user to fill in.
"""
            with st.spinner("Generating document..."):
                doc_result = ask_gemini(prompt)
            st.markdown(doc_result)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Thesis AI Agent prototype | Gemini for AI analysis | OpenAlex for "
    "academic discovery | Verify all sources before citing them in a thesis."
)