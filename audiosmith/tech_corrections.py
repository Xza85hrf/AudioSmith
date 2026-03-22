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

# Spanish suffix — handles plural forms in Spanish (minimal; Spanish doesn't decline like Polish).
# e.g. "Dockers" (plural), "el Docker" (with article)
ES_SUFFIX = r'(?:es|s)?'

# German grammatical case suffix — handles declension of tech terms in German speech.
# e.g. "des Dockers" (genitive), "dem Docker" (dative), "den Docker" (accusative), "Dockers" (plural)
DE_SUFFIX = r'(?:s|er|ern|es)?'

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

DE_PATTERNS: List[Tuple[str, str]] = [
    # ── Containers & orchestration ──
    (rf'\b(?:doker|dokker|docker){DE_SUFFIX}\b', 'Docker'),
    (r'\bdocker\s?compose\b', 'Docker Compose'),
    (rf'\b(?:kubernets|kubernetes){DE_SUFFIX}\b', 'Kubernetes'),
    (r'\bk8s\b', 'Kubernetes'),
    (rf'\b(?:konteiner|container){DE_SUFFIX}\b', 'Container'),

    # ── Version control ──
    (rf'\b(?:github|githab|gith[ua]b){DE_SUFFIX}\b', 'GitHub'),
    (rf'\b(?:gitlab|gitlab){DE_SUFFIX}\b', 'GitLab'),
    (rf'\b(?:bitbucket){DE_SUFFIX}\b', 'Bitbucket'),
    (rf'\b(?:pull\s?request|pul\s?rekwest){DE_SUFFIX}\b', 'pull request'),
    (rf'\b(?:merge|mertsch){DE_SUFFIX}\b', 'merge'),
    (rf'\b(?:commit|komit){DE_SUFFIX}\b', 'commit'),
    (rf'\b(?:branch|bränch|brentsch){DE_SUFFIX}\b', 'branch'),

    # ── Programming languages ──
    (rf'\b(?:dschawaskript|java\s?script|javascript){DE_SUFFIX}\b', 'JavaScript'),
    (rf'\b(?:taijpskript|taipskript|typescript){DE_SUFFIX}\b', 'TypeScript'),
    (rf'\b(?:paiton|python){DE_SUFFIX}\b', 'Python'),
    (rf'\b(?:dschawa|java){DE_SUFFIX}\b', 'Java'),
    (rf'\b(?:tschhärp|c\s?scharp|c#){DE_SUFFIX}\b', 'C#'),
    (rf'\b(?:c\s?plus\s?plus|cplusplus){DE_SUFFIX}\b', 'C++'),
    (rf'\b(?:rascht|rust){DE_SUFFIX}\b', 'Rust'),
    (rf'\b(?:robi|ruby){DE_SUFFIX}\b', 'Ruby'),
    (rf'\b(?:golang|go){DE_SUFFIX}\b', 'Go'),
    (rf'\b(?:php){DE_SUFFIX}\b', 'PHP'),
    (rf'\b(?:swift|swieft){DE_SUFFIX}\b', 'Swift'),
    (rf'\b(?:kotlin){DE_SUFFIX}\b', 'Kotlin'),

    # ── Frameworks & libraries ──
    (rf'\b(?:riäkt|react){DE_SUFFIX}\b', 'React'),
    (rf'\b(?:vue|wju){DE_SUFFIX}\b', 'Vue.js'),
    (r'\bvue\.?js\b', 'Vue.js'),
    (rf'\b(?:next|neckst){DE_SUFFIX}\b', 'Next.js'),
    (r'\bnext\.?js\b', 'Next.js'),
    (rf'\b(?:noud|node\.?js){DE_SUFFIX}\b', 'Node.js'),
    (rf'\b(?:django|dschango){DE_SUFFIX}\b', 'Django'),
    (r'\bdjango\b', 'Django'),
    (rf'\b(?:fastapi|fastäpi){DE_SUFFIX}\b', 'FastAPI'),
    (r'\bfastapi\b', 'FastAPI'),
    (rf'\b(?:angular){DE_SUFFIX}\b', 'Angular'),
    (rf'\b(?:spring\s?boot){DE_SUFFIX}\b', 'Spring Boot'),
    (rf'\b(?:asp\.?net|äspnett){DE_SUFFIX}\b', 'ASP.NET'),
    (rf'\b(?:\.net|dotnet){DE_SUFFIX}\b', '.NET'),
    (rf'\b(?:jquery|dschey\s?query){DE_SUFFIX}\b', 'jQuery'),
    (rf'\b(?:bootstrap|butstrap){DE_SUFFIX}\b', 'Bootstrap'),
    (rf'\b(?:tailwind|tejlwind){DE_SUFFIX}\b', 'Tailwind'),
    (rf'\b(?:svelte){DE_SUFFIX}\b', 'Svelte'),
    (rf'\b(?:laravel){DE_SUFFIX}\b', 'Laravel'),
    (rf'\b(?:flask){DE_SUFFIX}\b', 'Flask'),

    # ── Databases ──
    # NOTE: PostgreSQL pattern MUST come before SQL to avoid partial matching
    (rf'\b(?:postgre\s?sikjuel|postgresql){DE_SUFFIX}\b', 'PostgreSQL'),
    (rf'\b(?:sikjuel|sql){DE_SUFFIX}\b', 'SQL'),
    (rf'\b(?:mongo\s?di\s?bi|mongodb){DE_SUFFIX}\b', 'MongoDB'),
    (rf'\b(?:majsikjuel|mysql){DE_SUFFIX}\b', 'MySQL'),
    (rf'\b(?:redis){DE_SUFFIX}\b', 'Redis'),
    (rf'\b(?:elasticsearch){DE_SUFFIX}\b', 'Elasticsearch'),
    (rf'\b(?:sqlite|sikjuel\s?ait){DE_SUFFIX}\b', 'SQLite'),
    (rf'\b(?:mariadb|maria\s?di\s?bi){DE_SUFFIX}\b', 'MariaDB'),
    (rf'\b(?:cassandra|kassandra){DE_SUFFIX}\b', 'Cassandra'),
    (rf'\b(?:oracle){DE_SUFFIX}\b', 'Oracle'),
    (rf'\b(?:dynamodb){DE_SUFFIX}\b', 'DynamoDB'),

    # ── Cloud & infrastructure ──
    (rf'\b(?:rest\s?äpi|rest\s?api){DE_SUFFIX}\b', 'REST API'),
    (rf'\b(?:äpi|api){DE_SUFFIX}\b', 'API'),
    (rf'\b(?:ä\s?wu\s?es|aws){DE_SUFFIX}\b', 'AWS'),
    (rf'\b(?:azure|äschur){DE_SUFFIX}\b', 'Azure'),
    (r'\bazure\b', 'Azure'),
    (rf'\b(?:dschee\s?see\s?pi|gcp){DE_SUFFIX}\b', 'GCP'),
    (rf'\b(?:terraform){DE_SUFFIX}\b', 'Terraform'),
    (r'\bterraform\b', 'Terraform'),
    (rf'\b(?:ansible){DE_SUFFIX}\b', 'Ansible'),
    (r'\bansible\b', 'Ansible'),
    (rf'\b(?:nginx|ändschinäks){DE_SUFFIX}\b', 'Nginx'),
    (rf'\b(?:apache){DE_SUFFIX}\b', 'Apache'),
    (rf'\b(?:vercel|wärsäl){DE_SUFFIX}\b', 'Vercel'),
    (rf'\b(?:heroku){DE_SUFFIX}\b', 'Heroku'),

    # ── DevOps & CI/CD ──
    (rf'\b(?:si\s?aj\s?si\s?di|cicd|ci\s?cd){DE_SUFFIX}\b', 'CI/CD'),
    (r'\bdevops\b', 'DevOps'),
    (rf'\b(?:dschjenkins|jenkins){DE_SUFFIX}\b', 'Jenkins'),
    (r'\bjenkins\b', 'Jenkins'),
    (rf'\b(?:pipelinepaipläin){DE_SUFFIX}\b', 'pipeline'),
    (rf'\b(?:deployment|depläjment){DE_SUFFIX}\b', 'deployment'),
    (rf'\b(?:prometheus|prometeus){DE_SUFFIX}\b', 'Prometheus'),

    # ── Networking & protocols ──
    # NOTE: HTTPS forms must come before HTTP to avoid partial matching
    (r'\bäitsch\s+tipi\s+tipi\s+(?:äs|es)\b', 'HTTPS'),
    (r'\bhttps\b', 'HTTPS'),
    (r'\bäitsch\s+tipi\s+tipi\b', 'HTTP'),
    (r'\bhttp\b', 'HTTP'),
    (rf'\b(?:tipi\s?sipi|tcp){DE_SUFFIX}\b', 'TCP'),
    (rf'\b(?:di\s?äns|dns){DE_SUFFIX}\b', 'DNS'),
    (r'\b(?:es\s+es\s+ha|äss\s+äss\s+ha|ssh)\b', 'SSH'),
    (rf'\b(?:rdp){DE_SUFFIX}\b', 'RDP'),
    (rf'\b(?:äss\s?äss\s?äl|ssl){DE_SUFFIX}\b', 'SSL'),
    (rf'\b(?:websocket|web\s?soket){DE_SUFFIX}\b', 'WebSocket'),
    (rf'\b(?:graphql|graf\s?kju\s?el){DE_SUFFIX}\b', 'GraphQL'),
    (rf'\b(?:grpc|dschee\s?ar\s?pipi\s?si){DE_SUFFIX}\b', 'gRPC'),
    (rf'\b(?:cors){DE_SUFFIX}\b', 'CORS'),
    (rf'\b(?:oauth|o\s?auth){DE_SUFFIX}\b', 'OAuth'),
    (r'\budp\b', 'UDP'),
    (r'\btls\b', 'TLS'),

    # ── Architecture & patterns ──
    (rf'\b(?:bäkent|backend){DE_SUFFIX}\b', 'backend'),
    (rf'\b(?:frontänd|frontend){DE_SUFFIX}\b', 'frontend'),
    (rf'\b(?:middleware|midlwär){DE_SUFFIX}\b', 'middleware'),
    (rf'\b(?:serverles|serverless){DE_SUFFIX}\b', 'serverless'),
    (rf'\b(?:webhook|web\s?huk){DE_SUFFIX}\b', 'webhook'),
    (rf'\b(?:endpoint|endpäjnt){DE_SUFFIX}\b', 'endpoint'),
    (rf'\b(?:full\s?stack|fullstak){DE_SUFFIX}\b', 'full stack'),

    # ── Methodology ──
    (rf'\b(?:agile|ädschail){DE_SUFFIX}\b', 'Agile'),
    (rf'\b(?:scrum|skriem){DE_SUFFIX}\b', 'Scrum'),
    (rf'\b(?:standup|ständäp){DE_SUFFIX}\b', 'standup'),

    # ── Testing ──
    (rf'\b(?:cypress|zaipriss){DE_SUFFIX}\b', 'Cypress'),
    (rf'\b(?:unittest|unjut\s?test){DE_SUFFIX}\b', 'unit test'),
    (rf'\b(?:selenium|selenium){DE_SUFFIX}\b', 'Selenium'),

    # ── Build tools & package managers ──
    (rf'\b(?:npm|ändschäm|en\s?pi\s?em){DE_SUFFIX}\b', 'npm'),
    (rf'\b(?:webpack|webb\s?päck){DE_SUFFIX}\b', 'webpack'),
    (rf'\b(?:maven|mäwen){DE_SUFFIX}\b', 'Maven'),
    (rf'\b(?:gradle|grädle){DE_SUFFIX}\b', 'Gradle'),
    (r'\byarn\b', 'Yarn'),
    (r'\bpip\b', 'pip'),

    # ── Security & auth ──
    (rf'\b(?:jwt|dschej\s?dablju\s?ti){DE_SUFFIX}\b', 'JWT'),
    (rf'\b(?:firewall|fajrläl){DE_SUFFIX}\b', 'firewall'),

    # ── Linux/system commands ──
    (rf'\b(?:chmod|tschmot){DE_SUFFIX}\b', 'chmod'),
    (rf'\b(?:garbage\s?collector|gärbätsch\s?källektor){DE_SUFFIX}\b', 'garbage collector'),

    # ── General dev terms ──
    (rf'\b(?:framework|fräjmwork){DE_SUFFIX}\b', 'framework'),
    (rf'\b(?:library|läibräri){DE_SUFFIX}\b', 'library'),
    (rf'\b(?:cache|käsch){DE_SUFFIX}\b', 'cache'),
    (rf'\b(?:linux|linuks){DE_SUFFIX}\b', 'Linux'),
    (rf'\b(?:windows|windouws){DE_SUFFIX}\b', 'Windows'),
    (r'\bwindows\b', 'Windows'),
    (rf'\b(?:vs\s?code|viäss\s?koud){DE_SUFFIX}\b', 'VS Code'),
    (rf'\b(?:intellij|intälidschej){DE_SUFFIX}\b', 'IntelliJ'),
    (rf'\b(?:powershell|pauerräschäl){DE_SUFFIX}\b', 'PowerShell'),
    (rf'\b(?:bash|bäsch){DE_SUFFIX}\b', 'bash'),
    (r'\bmacos\b', 'macOS'),

    # ── ML & Data ──
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
]

ES_PATTERNS: List[Tuple[str, str]] = [
    # ── Containers & orchestration ──
    (rf'\b(?:doquer|dóker|docker){ES_SUFFIX}\b', 'Docker'),
    (r'\bdocker\s?compose\b', 'Docker Compose'),
    (rf'\b(?:cubernetis|kubernetes){ES_SUFFIX}\b', 'Kubernetes'),
    (r'\bk8s\b', 'Kubernetes'),
    (rf'\bcontenedor{ES_SUFFIX}\b', 'contenedor'),

    # ── Version control ──
    (rf'\b(?:guijab|gitjab|github){ES_SUFFIX}\b', 'GitHub'),
    (rf'\b(?:gitlab){ES_SUFFIX}\b', 'GitLab'),
    (rf'\b(?:pull\s?request|solicitud\s?de\s?extracción){ES_SUFFIX}\b', 'pull request'),
    (rf'\bfusionar{ES_SUFFIX}\b', 'merge'),
    (rf'\bconfirmación{ES_SUFFIX}\b', 'commit'),
    (rf'\b(?:rama|branch){ES_SUFFIX}\b', 'branch'),

    # ── Programming languages ──
    (rf'\b(?:yavascript|yabascript|jávaskrip|javascript){ES_SUFFIX}\b', 'JavaScript'),
    (rf'\b(?:taipscript|taipescript|typescript){ES_SUFFIX}\b', 'TypeScript'),
    (rf'\b(?:paiton|paitón|python){ES_SUFFIX}\b', 'Python'),
    (rf'\b(?:java|jaba){ES_SUFFIX}\b', 'Java'),
    (rf'\b(?:c\s?sharp|cicharp){ES_SUFFIX}\b', 'C#'),
    (rf'\b(?:c\s?plus\s?plus|ciplusplus){ES_SUFFIX}\b', 'C++'),
    (rf'\brrust{ES_SUFFIX}\b', 'Rust'),
    (rf'\brubí{ES_SUFFIX}\b', 'Ruby'),
    (rf'\bgolang{ES_SUFFIX}\b', 'Go'),
    (rf'\b(?:p\s?h\s?p|php){ES_SUFFIX}\b', 'PHP'),

    # ── Frameworks & libraries ──
    (rf'\b(?:riact|react){ES_SUFFIX}\b', 'React'),
    (rf'\b(?:vue|vu\.?js){ES_SUFFIX}\b', 'Vue.js'),
    (rf'\b(?:next\.?js|siguiente\.?js){ES_SUFFIX}\b', 'Next.js'),
    (rf'\b(?:node\.?js|nod\s?jés){ES_SUFFIX}\b', 'Node.js'),
    (rf'\bfastapi{ES_SUFFIX}\b', 'FastAPI'),
    (rf'\bdjango{ES_SUFFIX}\b', 'Django'),
    (rf'\b(?:spring\s?boot|springboot){ES_SUFFIX}\b', 'Spring Boot'),
    (rf'\b(?:asp\.?net|aspeenet){ES_SUFFIX}\b', 'ASP.NET'),
    (rf'\b(?:\.net|dotnet){ES_SUFFIX}\b', '.NET'),
    (rf'\bjquery{ES_SUFFIX}\b', 'jQuery'),
    (rf'\bbootstrap{ES_SUFFIX}\b', 'Bootstrap'),
    (rf'\btailwind{ES_SUFFIX}\b', 'Tailwind'),
    (rf'\bsvelte{ES_SUFFIX}\b', 'Svelte'),
    (rf'\blaravel{ES_SUFFIX}\b', 'Laravel'),
    (rf'\bflask{ES_SUFFIX}\b', 'Flask'),
    (rf'\bangular{ES_SUFFIX}\b', 'Angular'),

    # ── Databases ──
    (rf'\b(?:sql|sqel){ES_SUFFIX}\b', 'SQL'),
    (rf'\b(?:postgre|postgresql){ES_SUFFIX}\b', 'PostgreSQL'),
    (rf'\b(?:mongui|mongodb){ES_SUFFIX}\b', 'MongoDB'),
    (rf'\b(?:mysql|maisquél){ES_SUFFIX}\b', 'MySQL'),
    (rf'\bredis{ES_SUFFIX}\b', 'Redis'),
    (rf'\belasticsearch{ES_SUFFIX}\b', 'Elasticsearch'),
    (rf'\b(?:sqlite|esquelita){ES_SUFFIX}\b', 'SQLite'),
    (rf'\bmariadb{ES_SUFFIX}\b', 'MariaDB'),
    (rf'\bcassandra{ES_SUFFIX}\b', 'Cassandra'),
    (rf'\boracle{ES_SUFFIX}\b', 'Oracle'),
    (rf'\bdynamodb{ES_SUFFIX}\b', 'DynamoDB'),

    # ── Cloud & infrastructure ──
    (rf'\b(?:rest\s?api|api\s?rest){ES_SUFFIX}\b', 'REST API'),
    (rf'\bapi{ES_SUFFIX}\b', 'API'),
    (rf'\baws{ES_SUFFIX}\b', 'AWS'),
    (rf'\bazure{ES_SUFFIX}\b', 'Azure'),
    (rf'\bgcp{ES_SUFFIX}\b', 'GCP'),
    (rf'\bterraform{ES_SUFFIX}\b', 'Terraform'),
    (rf'\bansible{ES_SUFFIX}\b', 'Ansible'),
    (rf'\bnginx{ES_SUFFIX}\b', 'Nginx'),
    (rf'\bapache{ES_SUFFIX}\b', 'Apache'),
    (rf'\bvercel{ES_SUFFIX}\b', 'Vercel'),
    (rf'\b(?:load\s?balancer|balanceador\s?de\s?carga){ES_SUFFIX}\b', 'load balancer'),
    (rf'\bheroku{ES_SUFFIX}\b', 'Heroku'),

    # ── DevOps & CI/CD ──
    (rf'\b(?:ci\s?[/]?\s?cd|cicd){ES_SUFFIX}\b', 'CI/CD'),
    (rf'\bdevops{ES_SUFFIX}\b', 'DevOps'),
    (rf'\bjenkins{ES_SUFFIX}\b', 'Jenkins'),
    (rf'\bpipeline{ES_SUFFIX}\b', 'pipeline'),
    (rf'\b(?:deployment|implementación){ES_SUFFIX}\b', 'deployment'),
    (rf'\bprometheus{ES_SUFFIX}\b', 'Prometheus'),
    (r'\bcircleci\b', 'CircleCI'),

    # ── Networking & protocols ──
    (rf'\b(?:https|jétis){ES_SUFFIX}\b', 'HTTPS'),
    (rf'\b(?:http|jét){ES_SUFFIX}\b', 'HTTP'),
    (rf'\b(?:tcp|ti\s?ce\s?pe){ES_SUFFIX}\b', 'TCP'),
    (rf'\b(?:dns|denes){ES_SUFFIX}\b', 'DNS'),
    (rf'\b(?:ssh|ese\s?ese\s?jache){ES_SUFFIX}\b', 'SSH'),
    (rf'\brdp{ES_SUFFIX}\b', 'RDP'),
    (rf'\b(?:ssl|ese\s?ese\s?ele){ES_SUFFIX}\b', 'SSL'),
    (rf'\bwebsocket{ES_SUFFIX}\b', 'WebSocket'),
    (rf'\bgraphql{ES_SUFFIX}\b', 'GraphQL'),
    (rf'\bgrpc{ES_SUFFIX}\b', 'gRPC'),
    (rf'\bcors{ES_SUFFIX}\b', 'CORS'),
    (rf'\boauth{ES_SUFFIX}\b', 'OAuth'),
    (r'\budp\b', 'UDP'),
    (r'\btls\b', 'TLS'),

    # ── Architecture & patterns ──
    (rf'\b(?:backend){ES_SUFFIX}\b', 'backend'),
    (rf'\bfrontend{ES_SUFFIX}\b', 'frontend'),
    (rf'\bmiddleware{ES_SUFFIX}\b', 'middleware'),
    (rf'\bserverless{ES_SUFFIX}\b', 'serverless'),
    (rf'\bwebhook{ES_SUFFIX}\b', 'webhook'),
    (rf'\bendpoint{ES_SUFFIX}\b', 'endpoint'),
    (rf'\b(?:full\s?stack|pila\s?completa){ES_SUFFIX}\b', 'full stack'),

    # ── Methodology ──
    (rf'\b(?:agile|ágil){ES_SUFFIX}\b', 'Agile'),
    (rf'\bscrum{ES_SUFFIX}\b', 'Scrum'),
    (rf'\bstandup{ES_SUFFIX}\b', 'standup'),

    # ── Testing ──
    (rf'\bcypress{ES_SUFFIX}\b', 'Cypress'),
    (rf'\b(?:unit\s?test|prueba\s?unitaria){ES_SUFFIX}\b', 'unit test'),
    (rf'\bselenium{ES_SUFFIX}\b', 'Selenium'),

    # ── Build tools & package managers ──
    (rf'\bnpm{ES_SUFFIX}\b', 'npm'),
    (rf'\bwebpack{ES_SUFFIX}\b', 'webpack'),
    (rf'\bmaven{ES_SUFFIX}\b', 'Maven'),
    (rf'\bgradle{ES_SUFFIX}\b', 'Gradle'),
    (r'\byarn\b', 'Yarn'),
    (r'\bpip\b', 'pip'),

    # ── Security & auth ──
    (rf'\b(?:jwt|jota\s?doble\s?ve\s?te){ES_SUFFIX}\b', 'JWT'),
    (rf'\bfirewall{ES_SUFFIX}\b', 'firewall'),

    # ── Linux/system commands ──
    (rf'\bchmod{ES_SUFFIX}\b', 'chmod'),
    (rf'\b(?:garbage\s?collector|recolector\s?de\s?basura){ES_SUFFIX}\b', 'garbage collector'),
    (rf'\b(?:use\s?case|caso\s?de\s?uso){ES_SUFFIX}\b', 'use case'),
    (rf'\bextend{ES_SUFFIX}\b', 'extend'),

    # ── General dev terms ──
    (rf'\bframework{ES_SUFFIX}\b', 'framework'),
    (rf'\blibrary{ES_SUFFIX}\b', 'library'),
    (rf'\bcache{ES_SUFFIX}\b', 'cache'),
    (rf'\b(?:linux|linus){ES_SUFFIX}\b', 'Linux'),
    (rf'\bwindows{ES_SUFFIX}\b', 'Windows'),
    (rf'\bvs\s?code{ES_SUFFIX}\b', 'VS Code'),
    (rf'\b(?:intellij|inteliyé){ES_SUFFIX}\b', 'IntelliJ'),
    (rf'\bpowershell{ES_SUFFIX}\b', 'PowerShell'),
    (rf'\bbash{ES_SUFFIX}\b', 'bash'),
    (r'\bmacos\b', 'macOS'),

    # ── ML & Data ──
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
            # Map language codes to their pattern lists
            language_patterns: dict[str, List[Tuple[str, str]]] = {
                'pl': DEFAULT_PATTERNS,
                'de': DE_PATTERNS,
                'es': ES_PATTERNS,
            }
            patterns = language_patterns.get(self._language, DEFAULT_PATTERNS)
            for pattern, replacement in patterns:
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
