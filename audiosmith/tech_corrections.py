"""Tech term corrections — regex patterns for consistent technical terminology in transcripts.

Includes Polish phonetic Whisper corrections ported from InterviewCopilot.
Polish speech recognition (Whisper) phonetically transcribes English tech words as Polish text,
e.g. "Docker" → "dokier", "JavaScript" → "dżawaskrypt". These patterns fix that.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Polish grammatical case suffix — handles declension of tech terms in Polish speech.
# e.g. "Dockera" (genitive), "Dockerem" (instrumental), "Dockerze" (locative)
PL_SUFFIX = r'(?:a|em|owi|ze|ów|ami|ie|ę|ą)?'

DEFAULT_PATTERNS: List[Tuple[str, str]] = [
    # ── Containers & orchestration ──
    (r'\bdocker\s?compose\b', 'Docker Compose'),
    (rf'\b(?:joker|dokier|dokar[zż]?|do\s?karton|docker){PL_SUFFIX}\b', 'Docker'),
    (rf'\b(?:kubernets|kubernet[eé]s|kubernetes){PL_SUFFIX}\b', 'Kubernetes'),
    (r'\bk8s\b', 'Kubernetes'),
    (rf'\bkontajner{PL_SUFFIX}\b', 'kontener'),

    # ── Version control ──
    (rf'\b(?:bit\s?bucket){PL_SUFFIX}\b', 'Bitbucket'),
    (rf'\b(?:git\s?hab|github){PL_SUFFIX}\b', 'GitHub'),
    (rf'\b(?:git\s?lab|gitlab){PL_SUFFIX}\b', 'GitLab'),
    (rf'\b(?:pul\s?rekłest){PL_SUFFIX}\b', 'pull request'),
    (rf'\bmerdż{PL_SUFFIX}\b', 'merge'),
    (rf'\bkomit{PL_SUFFIX}\b', 'commit'),
    (rf'\b(?:bra[ńn]cz){PL_SUFFIX}\b', 'branch'),

    # ── Programming languages ──
    (rf'\b(?:dżawaskrypt|jawaskrypt|jaw[aą]\s?skrypt|javascript){PL_SUFFIX}\b', 'JavaScript'),
    (rf'\b(?:tajpskrypt|tajp\s?skrypt|typescript){PL_SUFFIX}\b', 'TypeScript'),
    (rf'\bpaj?ton{PL_SUFFIX}\b', 'Python'),
    (r'\bpython\b', 'Python'),
    (rf'\bdżaw[aęy]?{PL_SUFFIX}\b', 'Java'),
    (rf'\b(?:si\s?szarp|c\s?szarp){PL_SUFFIX}\b', 'C#'),
    (rf'\bsi\s?plus\s?plus{PL_SUFFIX}\b', 'C++'),
    (rf'\brast{PL_SUFFIX}\b', 'Rust'),
    (rf'\brubi{PL_SUFFIX}\b', 'Ruby'),
    (rf'\bgolang{PL_SUFFIX}\b', 'Go'),
    (rf'\b(?:pi\s?ejcz\s?pi){PL_SUFFIX}\b', 'PHP'),
    (rf'\bsłift{PL_SUFFIX}\b', 'Swift'),
    (rf'\bkotlin{PL_SUFFIX}\b', 'Kotlin'),

    # ── Frameworks & libraries ──
    (rf'\b(?:riakt|riyakt|react){PL_SUFFIX}\b', 'React'),
    (rf'\b(?:wju(?:\s?j[eé]?s?)?|vue){PL_SUFFIX}\b', 'Vue.js'),
    (rf'\b(?:nekst(?:\s?j[eé]?s?)?){PL_SUFFIX}\b', 'Next.js'),
    (r'\bnext\.?js\b', 'Next.js'),
    (rf'\b(?:not\s?dżi\s?es|nod\s?dżi\s?es|node\.?js){PL_SUFFIX}\b', 'Node.js'),
    (rf'\bfastap[iy]{PL_SUFFIX}\b', 'FastAPI'),
    (r'\bfastapi\b', 'FastAPI'),
    (rf'\bdżango{PL_SUFFIX}\b', 'Django'),
    (r'\bdjango\b', 'Django'),
    (rf'\b(?:spring\s?b[uo]{{1,2}}t){PL_SUFFIX}\b', 'Spring Boot'),
    (rf'\b(?:aspodnet|asbot\s?net|asp\s?o?d?\s?net|psp){PL_SUFFIX}\b', 'ASP.NET'),
    (rf'\b(?:dotnet|dot\s?net){PL_SUFFIX}\b', '.NET'),
    (rf'\b(?:dżej\s?kłeri){PL_SUFFIX}\b', 'jQuery'),
    (rf'\b(?:butstrap){PL_SUFFIX}\b', 'Bootstrap'),
    (rf'\b(?:tejlłind|tejl\s?łind){PL_SUFFIX}\b', 'Tailwind'),
    (rf'\bswelt{PL_SUFFIX}\b', 'Svelte'),
    (rf'\blaravel{PL_SUFFIX}\b', 'Laravel'),
    (rf'\bflask{PL_SUFFIX}\b', 'Flask'),
    (r'\bangular\b', 'Angular'),

    # ── Databases ──
    (rf'\b(?:sikłel|siku\s?el|sql){PL_SUFFIX}\b', 'SQL'),
    (rf'\b(?:postgre(?:s|sy|siku|sikłel)?|postgresql){PL_SUFFIX}\b', 'PostgreSQL'),
    (rf'\b(?:mongi?\s?di?\s?bi?|mongodb){PL_SUFFIX}\b', 'MongoDB'),
    (rf'\b(?:maj\s?(?:sikłel|siku\s?el)|mysql){PL_SUFFIX}\b', 'MySQL'),
    (rf'\bredis{PL_SUFFIX}\b', 'Redis'),
    (rf'\b(?:el[ae]stik\s?s[eö]r[cč]|elasticsearch){PL_SUFFIX}\b', 'Elasticsearch'),
    (rf'\b(?:siku\s?lajt|sikłelajt|sqlite){PL_SUFFIX}\b', 'SQLite'),
    (rf'\b(?:maria\s?di\s?bi){PL_SUFFIX}\b', 'MariaDB'),
    (rf'\bkasandra{PL_SUFFIX}\b', 'Cassandra'),
    (rf'\borakle{PL_SUFFIX}\b', 'Oracle'),
    (rf'\b(?:dinamo\s?di\s?bi|dynamodb){PL_SUFFIX}\b', 'DynamoDB'),

    # ── Cloud & infrastructure ──
    (rf'\b(?:rest\s?(?:aj\s?pi|e?pi|api)){PL_SUFFIX}\b', 'REST API'),
    (rf'\b(?:aj\s?pi|api){PL_SUFFIX}\b', 'API'),
    (rf'\b(?:ej\s?dablju\s?es|a\s?wu\s?es|aws){PL_SUFFIX}\b', 'AWS'),
    (rf'\bażur{PL_SUFFIX}\b', 'Azure'),
    (r'\bazure\b', 'Azure'),
    (rf'\b(?:dżi\s?si\s?pi|gcp){PL_SUFFIX}\b', 'GCP'),
    (rf'\bteraform{PL_SUFFIX}\b', 'Terraform'),
    (r'\bterraform\b', 'Terraform'),
    (rf'\bansibel{PL_SUFFIX}\b', 'Ansible'),
    (r'\bansible\b', 'Ansible'),
    (rf'\b(?:en\s?dżi\s?en\s?[iy]ks|endżineks|nginx){PL_SUFFIX}\b', 'Nginx'),
    (rf'\bapacze{PL_SUFFIX}\b', 'Apache'),
    (rf'\bwersel{PL_SUFFIX}\b', 'Vercel'),
    (rf'\b(?:load|lod|long|lot)\s?(?:balan[sc][eao]?r?|balance|baransa?){PL_SUFFIX}\b',
     'load balancer'),
    (rf'\bheroku{PL_SUFFIX}\b', 'Heroku'),

    # ── DevOps & CI/CD ──
    (rf'\b(?:ci\s?[/]?\s?c[id]\s?d[iy]?|ci\s?[/]?\s?cd|si\s?aj\s?si\s?di|CACD){PL_SUFFIX}\b',
     'CI/CD'),
    (r'\bdevops\b', 'DevOps'),
    (rf'\bdżenkins{PL_SUFFIX}\b', 'Jenkins'),
    (r'\bjenkins\b', 'Jenkins'),
    (rf'\bpajplajn{PL_SUFFIX}\b', 'pipeline'),
    (rf'\b(?:depłoj|depłojment){PL_SUFFIX}\b', 'deployment'),
    (rf'\bprometeusz{PL_SUFFIX}\b', 'Prometheus'),
    (r'\bmikro\s?serwis[yów]*\b', 'mikroserwis'),
    (r'\bcircleci\b', 'CircleCI'),

    # ── Networking & protocols ──
    (rf'\b(?:ejcz\s?ti\s?ti\s?pi\s?es|https){PL_SUFFIX}\b', 'HTTPS'),
    (rf'\b(?:ejcz\s?ti\s?ti\s?pi|http){PL_SUFFIX}\b', 'HTTP'),
    (rf'\b(?:ti\s?si\s?pi|tcp){PL_SUFFIX}\b', 'TCP'),
    (rf'\b(?:di\s?en\s?es|dns){PL_SUFFIX}\b', 'DNS'),
    (rf'\b(?:es\s?es\s?hajcz|es\s?es\s?h|ssh){PL_SUFFIX}\b', 'SSH'),
    (rf'\b(?:arde?pa?|ar\s?di\s?pi|rdp){PL_SUFFIX}\b', 'RDP'),
    (rf'\b(?:es\s?es\s?el|ssl){PL_SUFFIX}\b', 'SSL'),
    (rf'\b(?:łeb\s?soket|web\s?soket|websocket){PL_SUFFIX}\b', 'WebSocket'),
    (rf'\b(?:graf\s?kju\s?el|graphql){PL_SUFFIX}\b', 'GraphQL'),
    (rf'\b(?:dżi\s?ar\s?pi\s?si|grpc){PL_SUFFIX}\b', 'gRPC'),
    (rf'\bkors{PL_SUFFIX}\b', 'CORS'),
    (rf'\b(?:o\s?a[łw]t|o\s?a[łw]f|oauth){PL_SUFFIX}\b', 'OAuth'),
    (r'\budp\b', 'UDP'),
    (r'\btls\b', 'TLS'),

    # ── Architecture & patterns ──
    (rf'\b(?:bek\s?end|bakend){PL_SUFFIX}\b', 'backend'),
    (rf'\b(?:front\s?end){PL_SUFFIX}\b', 'frontend'),
    (rf'\bmidłer{PL_SUFFIX}\b', 'middleware'),
    (rf'\b(?:serwer\s?les|serwerles){PL_SUFFIX}\b', 'serverless'),
    (rf'\b(?:łeb\s?huk|web\s?huk){PL_SUFFIX}\b', 'webhook'),
    (rf'\b(?:endpojnt|end\s?pojnt){PL_SUFFIX}\b', 'endpoint'),
    (rf'\b(?:ful\s?stak|full\s?stak){PL_SUFFIX}\b', 'full stack'),

    # ── Methodology ──
    (rf'\b(?:adżajl|edżajl){PL_SUFFIX}\b', 'Agile'),
    (rf'\bskram{PL_SUFFIX}\b', 'Scrum'),
    (rf'\bstendup{PL_SUFFIX}\b', 'standup'),

    # ── Testing ──
    (rf'\bsajpres{PL_SUFFIX}\b', 'Cypress'),
    (rf'\b(?:junit\s?test){PL_SUFFIX}\b', 'unit test'),
    (rf'\bselenium{PL_SUFFIX}\b', 'Selenium'),

    # ── Build tools & package managers ──
    (rf'\b(?:en\s?pi\s?em|npm){PL_SUFFIX}\b', 'npm'),
    (rf'\b(?:łeb\s?pak|łebpak|webpack){PL_SUFFIX}\b', 'webpack'),
    (rf'\bmejwen{PL_SUFFIX}\b', 'Maven'),
    (rf'\bgrejdel{PL_SUFFIX}\b', 'Gradle'),
    (r'\byarn\b', 'Yarn'),
    (r'\bpip\b', 'pip'),

    # ── Security & auth ──
    (rf'\b(?:dżej\s?(?:dablju|wu)\s?ti|jwt){PL_SUFFIX}\b', 'JWT'),
    (rf'\b(?:fajerwol|fajr?łol){PL_SUFFIX}\b', 'firewall'),

    # ── Linux/system commands ──
    (rf'\b(?:hamot[t]?|cz?h?mod){PL_SUFFIX}\b', 'chmod'),
    (rf'\b(?:garbycz|garbicz|garb[ie]dż)\s?kolekt[oe]r{PL_SUFFIX}\b', 'garbage collector'),
    (rf'\b(?:justys|ju[sz]\s?kejs){PL_SUFFIX}\b', 'use case'),
    (rf'\b(?:ekscent|eksten[td]){PL_SUFFIX}\b', 'extend'),

    # ── General dev terms ──
    (rf'\b(?:frejmłork|frejmwork){PL_SUFFIX}\b', 'framework'),
    (rf'\blajbreri{PL_SUFFIX}\b', 'library'),
    (rf'\bkesz{PL_SUFFIX}\b', 'cache'),
    (rf'\b(?:linuks|linkus[uo]?j?|linux){PL_SUFFIX}\b', 'Linux'),
    (rf'\bwindołs{PL_SUFFIX}\b', 'Windows'),
    (r'\bwindows\b', 'Windows'),
    (rf'\b(?:wi\s?es\s?kod){PL_SUFFIX}\b', 'VS Code'),
    (rf'\b(?:inteli\s?dżej){PL_SUFFIX}\b', 'IntelliJ'),
    (rf'\b(?:pałer\s?szel|pauer\s?szel){PL_SUFFIX}\b', 'PowerShell'),
    (rf'\bbasz{PL_SUFFIX}\b', 'bash'),
    (r'\bmacos\b', 'macOS'),

    # ── ML & Data (AudioSmith-specific) ──
    (r'\btensorflow\b', 'TensorFlow'),
    (r'\bpytorch\b', 'PyTorch'),
    (r'\bscikit[\s-]?learn\b', 'scikit-learn'),
    (r'\bnumpy\b', 'NumPy'),
    (r'\bpandas\b', 'Pandas'),

    # ── Data formats (uppercase conventions) ──
    (r'\bcrud\b', 'CRUD'),
    (r'\borm\b', 'ORM'),
    (r'\bcli\b', 'CLI'),
    (r'\bgui\b', 'GUI'),
    (r'\bjson\b', 'JSON'),
    (r'\bxml\b', 'XML'),
    (r'\byaml\b', 'YAML'),
    (r'\bcsv\b', 'CSV'),
    (r'\bhtml\b', 'HTML'),
    (r'\bcss\b', 'CSS'),

    # ── Polish Whisper verb corrections ──
    # Faster-Whisper drops prefixes or confuses verb forms in Polish.
    (r'\bwjaśnić\b', 'wyjaśnić'),
    (r'\bwjaśnij\b', 'wyjaśnij'),
    (r'\bwypisat\b', 'wypisać'),
    (r'\bopisat\b', 'opisać'),
    (r'\bwytłumaczyt\b', 'wytłumaczyć'),
    (r'\bopowiedziat\b', 'opowiedzieć'),

    # ── Whisper hallucination fixes ──
    # Common mishearings of tech terms and phrases in Polish speech
    (r'\bEICD\b', 'CI/CD'),
    (r'\bkontrolizacj', 'konteneryzacj'),
    (r'\bkrotenerezacj', 'konteneryzacj'),
    (r'\bpięcioletnie\b', 'pięć lat'),
    (r'\beBubble\b', 'w chmurze'),
    (r'\b(?:do\s?krzy)\b', 'w Dockerze'),
    (r'\bupitukarki\b', 'drukarki'),
    (r'\brony\b(?=\s*\.\|\s*$)', 'rano'),
    (r'\bCo to jest dobra\b', 'Co to jest Docker'),
    (r'\bJa zniczymy\b', 'Wyjaśnij czym jest'),
    (r'\bBadawski\b', ''),
    (r'\bPopatrz na moment\b', 'Opisz moment'),
    (r'\bprzestawia działać\b', 'przestaje działać'),
    (r'\bodszedłbyś\b', 'podszedłbyś'),
    (r'\bdokaże\b', 'Dockerze'),
    (r'\bautomatizacj', 'automatyzacj'),
    (r'\bopis\b(?=\s+(?:swoj[eaiąę]|swój|swoim|swoich|twoj[eaiąę]|twój|twoim|nam|mi|o\b))',
     'Opisz'),
    (r'^Dlatego chcę zmienić\b', 'Dlaczego chcesz zmienić'),
    (r'\bmowa prasę\b', 'umowa o pracę'),
    (r'\bco obiecał nasze firmy\b', 'co przyciąga Cię w naszej firmie'),
    (r'\bPopisz\b', 'Opisz'),
    (r'^To to jest\b', 'Co to jest'),
    (r'\bwyjdzie z siebie\b', 'widzisz siebie'),
    (r'\bzmienić roboproces\b', 'zmienić pracę'),
    (r'\bby się stał odpowiadać\b', 'przestał odpowiadać'),
]


class TechTermCorrections:
    """Apply regex-based corrections for technical terminology in transcripts.

    Handles both standard case-normalization (e.g. "javascript" → "JavaScript")
    and language-specific phonetic corrections from Whisper (e.g. Polish "dokier" → "Docker").

    Args:
        language: ISO 639-1 code. Patterns are loaded for languages that have them;
                  unsupported languages get an empty corrector (no-op).
    """

    def __init__(self, language: str = "pl") -> None:
        self._language = language
        self._patterns: List[Tuple[str, str, str]] = []
        self._compiled: List[Tuple[re.Pattern, str]] = []
        self._load_defaults()

    def _load_defaults(self) -> None:
        from audiosmith.language_data import get_language
        config = get_language(self._language)
        if config.has_tech_corrections:
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
