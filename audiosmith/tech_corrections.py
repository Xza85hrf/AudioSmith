"""Tech term corrections — regex patterns for consistent technical terminology in transcripts."""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_PATTERNS: List[Tuple[str, str]] = [
    # Programming Languages
    (r'\bjavascript\b', 'JavaScript'),
    (r'\btypescript\b', 'TypeScript'),
    (r'\bpython\b', 'Python'),
    (r'\bnode\.?js\b', 'Node.js'),
    (r'\bnextjs\b', 'Next.js'),
    (r'\bnext\.?js\b', 'Next.js'),
    (r'\breact\b', 'React'),
    (r'\bangular\b', 'Angular'),
    (r'\bvue\b', 'Vue'),
    (r'\bdjango\b', 'Django'),
    (r'\bflask\b', 'Flask'),
    (r'\bfastapi\b', 'FastAPI'),
    (r'\bspring\s?boot\b', 'Spring Boot'),
    (r'\bgolang\b', 'Go'),
    (r'\brusht\b', 'Rust'),
    (r'\bkotlin\b', 'Kotlin'),
    (r'\bswift\b', 'Swift'),
    # Databases
    (r'\bpostgresql\b', 'PostgreSQL'),
    (r'\bpostgres\b', 'PostgreSQL'),
    (r'\bmysql\b', 'MySQL'),
    (r'\bmongodb\b', 'MongoDB'),
    (r'\bmongo\b', 'MongoDB'),
    (r'\bredis\b', 'Redis'),
    (r'\belasticsearch\b', 'Elasticsearch'),
    (r'\bdynamodb\b', 'DynamoDB'),
    (r'\bsqlite\b', 'SQLite'),
    # Cloud & Infrastructure
    (r'\bkubernetes\b', 'Kubernetes'),
    (r'\bk8s\b', 'Kubernetes'),
    (r'\bdocker\b', 'Docker'),
    (r'\bterraform\b', 'Terraform'),
    (r'\bansible\b', 'Ansible'),
    # Version Control & CI
    (r'\bgithub\b', 'GitHub'),
    (r'\bgitlab\b', 'GitLab'),
    (r'\bbitbucket\b', 'Bitbucket'),
    (r'\bjenkins\b', 'Jenkins'),
    (r'\bcircleci\b', 'CircleCI'),
    # Protocols & Formats
    (r'\bgraphql\b', 'GraphQL'),
    (r'\bgrpc\b', 'gRPC'),
    (r'\bwebsocket\b', 'WebSocket'),
    (r'\boauth\b', 'OAuth'),
    # ML & Data
    (r'\btensorflow\b', 'TensorFlow'),
    (r'\bpytorch\b', 'PyTorch'),
    (r'\bscikit[\s-]?learn\b', 'scikit-learn'),
    (r'\bnumpy\b', 'NumPy'),
    (r'\bpandas\b', 'Pandas'),
    # Polish Whisper garbles (from InterviewCopilot)
    (r'\bDocker\s?(?:ze|ize)\b', 'Dockerize'),
    (r'\bGit\s?Hub\b', 'GitHub'),
    (r'\bRe(?:act|akt)\b', 'React'),
    (r'\bJava\s?Script\b', 'JavaScript'),
    (r'\bType\s?Script\b', 'TypeScript'),
    (r'\bPost?gres?\b', 'PostgreSQL'),
    (r'\bKuberneti?s\b', 'Kubernetes'),
    (r'\bw chmurze\b', 'w chmurze'),
    (r'\bmy?ikro[\s-]?serwisy?\b', 'mikroserwisy'),
    (r'\bapi\b', 'API'),
    (r'\bcrud\b', 'CRUD'),
    (r'\borm\b', 'ORM'),
    (r'\bcli\b', 'CLI'),
    (r'\bgui\b', 'GUI'),
    (r'\bsql\b', 'SQL'),
    (r'\bjson\b', 'JSON'),
    (r'\bxml\b', 'XML'),
    (r'\byaml\b', 'YAML'),
    (r'\bcsv\b', 'CSV'),
    (r'\bhtml\b', 'HTML'),
    (r'\bcss\b', 'CSS'),
    (r'\bhttp\b', 'HTTP'),
    (r'\bhttps\b', 'HTTPS'),
    (r'\bssh\b', 'SSH'),
    (r'\btcp\b', 'TCP'),
    (r'\budp\b', 'UDP'),
    (r'\bdns\b', 'DNS'),
    (r'\bjwt\b', 'JWT'),
    (r'\bssl\b', 'SSL'),
    (r'\btls\b', 'TLS'),
    (r'\baws\b', 'AWS'),
    (r'\bgcp\b', 'GCP'),
    (r'\bazure\b', 'Azure'),
    (r'\bnpm\b', 'npm'),
    (r'\byarn\b', 'Yarn'),
    (r'\bpip\b', 'pip'),
    (r'\blinux\b', 'Linux'),
    (r'\bwindows\b', 'Windows'),
    (r'\bmacos\b', 'macOS'),
]


class TechTermCorrections:
    """Apply regex-based corrections for technical terminology in transcripts."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, str, str]] = []
        self._compiled: List[Tuple[re.Pattern, str]] = []
        self._load_defaults()

    def _load_defaults(self):
        for pattern, replacement in DEFAULT_PATTERNS:
            self.add_pattern(pattern, replacement)

    def add_pattern(self, pattern: str, replacement: str, name: str = '') -> None:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._patterns.append((pattern, replacement, name or pattern))
            self._compiled.append((compiled, replacement))
        except re.error as e:
            logger.error("Invalid regex pattern %s: %s", pattern, e)

    def clear(self) -> None:
        self._patterns.clear()
        self._compiled.clear()

    def correct(self, text: str) -> str:
        result = text
        for compiled, replacement in self._compiled:
            result = compiled.sub(replacement, result)
        return result

    def get_pattern_count(self) -> int:
        return len(self._patterns)
