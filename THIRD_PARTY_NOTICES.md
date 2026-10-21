# Third-Party Notices

## LocalRAG — Third-Party Software and Model Attributions

LocalRAG is a local-first, open-source AI research assistant that uses third-party software libraries, machine learning models, and optional external metadata services.

Original LocalRAG source code is intended to be distributed under the **GNU Affero General Public License, Version 3 or later (AGPL-3.0-or-later)**.

See the root `LICENSE` file for the full license text.

Third-party components remain subject to their respective licenses. LocalRAG's project license does not replace, modify, or supersede those licenses.

---

## 1. Third-Party Software Libraries

### Direct Python and build dependencies

The `pyproject.toml` dependency groups currently declare:

- **PyMuPDF** (base runtime): AGPLv3 or Artifex commercial license; see the section below. This is the primary license compatibility issue for redistribution.
- **faiss-cpu** (`retrieval`, `rag`, `dashboard` extras): MIT.
- **NumPy** (embedding/retrieval/dashboard extras): BSD 3-Clause; its binary distributions include notices for bundled components.
- **sentence-transformers** (embedding/retrieval/rag/dashboard extras): Apache-2.0.
- **Streamlit** (`dashboard` extra): Apache-2.0.
- **pytest** (`dev` extra): MIT.
- **setuptools** and **wheel** (build system): MIT. Setuptools 77 or later is required for the SPDX license expression in `pyproject.toml`.

These packages are installed as separate dependencies; LocalRAG does not vendor their source or binaries. Their listed licenses are a summary, not a complete notice bundle for every transitive dependency. In particular, the Sentence Transformers stack also installs libraries such as PyTorch, Transformers, Hugging Face Hub, Tokenizers, SciPy, and scikit-learn. Review the actual versions and included notices when preparing a bundled distribution.

### PyMuPDF

- **Purpose:** PDF document processing, text extraction, and metadata extraction.
- **License:** GNU Affero General Public License v3 (AGPLv3) or commercial licensing.
- **Developer:** Artifex Software, Inc.
- **Repository:** https://github.com/pymupdf/PyMuPDF
- **License information:** https://pymupdf.io/licensing

PyMuPDF is used to extract textual content and metadata from PDF documents.

Its use and redistribution remain subject to the applicable PyMuPDF and MuPDF license terms.

### FAISS — Facebook AI Similarity Search

- **Purpose:** Vector indexing, nearest-neighbor search, and semantic retrieval.
- **License:** MIT License.
- **Developer:** Meta AI Research.
- **Repository:** https://github.com/facebookresearch/faiss
- **License:** https://github.com/facebookresearch/faiss/blob/main/LICENSE

FAISS is used to store and search numerical document embeddings through similarity-based retrieval.

### Sentence Transformers

- **Purpose:** Text embedding generation and semantic similarity.
- **License:** Apache License 2.0.
- **Repository:** https://github.com/huggingface/sentence-transformers
- **License:** https://github.com/huggingface/sentence-transformers/blob/main/LICENSE

Sentence Transformers provides the embedding interface used by LocalRAG to transform document passages and user queries into numerical vector representations.

### Streamlit

- **Purpose:** Local web dashboard and interactive user interface.
- **License:** Apache License 2.0.
- **Repository:** https://github.com/streamlit/streamlit
- **License:** https://github.com/streamlit/streamlit/blob/develop/LICENSE

Streamlit provides the interactive dashboard for document management, question answering, semantic citation search, and bibliography review.

### Ollama

- **Purpose:** Local large language model execution and inference.
- **License:** MIT License for the Ollama software.
- **Repository:** https://github.com/ollama/ollama
- **License:** https://github.com/ollama/ollama/blob/main/LICENSE

Ollama is an optional, separately installed local runtime called through its HTTP API. It is not a Python dependency and LocalRAG does not bundle the Ollama executable.

The Ollama software license is separate from the licenses governing models executed through Ollama.

### NumPy

- **Purpose:** Numerical operations and embedding array processing.
- **Primary license:** BSD 3-Clause License, with additional notices applicable to some distributed components.
- **Website:** https://numpy.org/
- **License information:** https://numpy.org/doc/stable/license.html

NumPy is used for numerical array handling and vector processing.

---

## 2. Machine Learning Models

### all-MiniLM-L6-v2

- **Purpose:** Semantic text embedding generation.
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **License:** Apache License 2.0.
- **Model card:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

LocalRAG uses this model to generate 384-dimensional text embeddings for semantic search.

Model weights are downloaded separately and cached locally. They are not included in the LocalRAG source repository.

Users remain responsible for complying with applicable model terms.

### Meta Llama 3.2

- **Purpose:** Local language generation and retrieval-augmented question answering.
- **Default configuration:** llama3.2:3b
- **License:** Meta Llama 3.2 Community License Agreement.
- **License agreement:** https://dev.meta.ai/llama/llama3_2/license

Llama 3.2 is developed by Meta and is subject to its own license agreement and acceptable use policy.

**Model attribution:**

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

**Built with Llama**

LocalRAG supports local inference through Ollama using Meta Llama 3.2.

Model weights are not distributed as part of the LocalRAG source repository. Users obtain models separately through their local model provider.

Where applicable, use or distribution of Llama materials or products containing them must comply with Meta's attribution, license, redistribution, and acceptable use requirements.

---

## 3. External Metadata Services

### Crossref

- **Purpose:** Optional academic bibliographic metadata lookup.
- **Website:** https://www.crossref.org/
- **API documentation:** https://www.crossref.org/documentation/retrieve-metadata/rest-api/

LocalRAG can optionally query Crossref to retrieve bibliographic information associated with scholarly publications.

Crossref is an external metadata service rather than a locally bundled software dependency.

Crossref requests are optional and require network access.

The accuracy, availability, and reuse rights of retrieved metadata remain subject to the relevant sources and applicable terms.

Crossref states that nearly all of its bibliographic metadata is reusable without restriction and that Crossref-generated data is CC0; abstracts may remain copyrighted by their publishers or authors. LocalRAG's optional lookup uses bibliographic metadata and does not download article full text. See [Crossref metadata licensing](https://www.crossref.org/documentation/retrieve-metadata/) before redistributing metadata or abstracts.

---

## 4. User Documents and Research Papers

LocalRAG allows users to upload, index, search, and analyze their own PDF and TXT documents.

Uploaded documents and indexed research papers are not automatically covered by LocalRAG's AGPL license.

Users are responsible for ensuring that they have the necessary rights to upload, process, redistribute, or publish their documents.

The public LocalRAG repository should not contain private research documents, proprietary papers, or copyrighted third-party content without appropriate permission.

---

## 5. Software and Model Distribution

LocalRAG is designed primarily for local execution.

The public source repository does not intentionally redistribute:

- Third-party pretrained model weights.
- Private user documents.
- Generated FAISS indexes.
- Locally saved bibliography approvals.
- User-specific application settings.
- Local credentials or secrets.

Users install Python dependencies and obtain supported AI models separately.

Any redistribution of third-party components, bundled binaries, or model weights must comply with their applicable license terms.

---

## 6. License Compliance

The original LocalRAG code is intended to be licensed under the GNU Affero General Public License, Version 3 or later.

Third-party components retain their original licenses and copyright notices.

This document is a summary of major third-party components and does not replace the complete applicable license agreements, copyright notices, or required redistribution materials.

Additional direct or transitive dependencies may carry their own license and notice requirements. These should be reviewed before distributing bundled software or binaries.

For more information about LocalRAG's licensing, refer to the `LICENSE` file in the project root.
