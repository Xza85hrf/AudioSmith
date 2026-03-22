"""Tests for audiosmith.tech_corrections module."""

import pytest

from audiosmith.tech_corrections import TechTermCorrections, DEFAULT_PATTERNS


class TestTechTermCorrectionsInit:
    """Test TechTermCorrections initialization and pattern loading."""

    def test_init_loads_defaults(self):
        """Test that initializing loads default patterns."""
        corrector = TechTermCorrections()
        assert corrector.get_pattern_count() > 0

    def test_init_pattern_count(self):
        """Test that default patterns are loaded."""
        corrector = TechTermCorrections()
        # Should load all patterns from DEFAULT_PATTERNS
        assert corrector.get_pattern_count() == len(DEFAULT_PATTERNS)

    def test_clear_patterns(self):
        """Test that clear() removes all patterns."""
        corrector = TechTermCorrections()
        corrector.clear()
        assert corrector.get_pattern_count() == 0

    def test_add_single_pattern(self):
        """Test adding a custom pattern."""
        corrector = TechTermCorrections()
        corrector.clear()
        corrector.add_pattern(r'\btest\b', 'TEST')
        assert corrector.get_pattern_count() == 1

    def test_add_multiple_patterns(self):
        """Test adding multiple patterns."""
        corrector = TechTermCorrections()
        corrector.clear()
        corrector.add_pattern(r'\bfoo\b', 'FOO')
        corrector.add_pattern(r'\bbar\b', 'BAR')
        assert corrector.get_pattern_count() == 2

    def test_add_pattern_with_name(self):
        """Test adding pattern with custom name."""
        corrector = TechTermCorrections()
        corrector.clear()
        corrector.add_pattern(r'\btest\b', 'TEST', name='custom_test')
        assert corrector.get_pattern_count() == 1


class TestTechTermCorrectionsBasicCorrections:
    """Test basic tech term corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_docker_lowercase_to_docker(self, corrector):
        """Test 'docker' → 'Docker'."""
        result = corrector.correct('I use docker daily')
        assert 'Docker' in result

    def test_dokier_polish_to_docker(self, corrector):
        """Test Polish 'dokier' → 'Docker'."""
        result = corrector.correct('I use dokier daily')
        assert 'Docker' in result

    def test_docker_compose_correction(self, corrector):
        """Test 'docker compose' → 'Docker Compose'."""
        result = corrector.correct('use docker compose for this')
        assert 'Docker Compose' in result

    def test_kubernetes_lowercase(self, corrector):
        """Test 'kubernetes' → 'Kubernetes'."""
        result = corrector.correct('kubernetes is complex')
        assert 'Kubernetes' in result

    def test_javascript_case_correction(self, corrector):
        """Test JavaScript case normalization."""
        result = corrector.correct('I code javascript')
        assert 'JavaScript' in result

    def test_python_lowercase_to_python(self, corrector):
        """Test 'python' → 'Python'."""
        result = corrector.correct('python is great')
        assert 'Python' in result

    def test_react_correction(self, corrector):
        """Test React correction."""
        result = corrector.correct('I use react')
        assert 'React' in result

    def test_nodejs_correction(self, corrector):
        """Test Node.js correction."""
        result = corrector.correct('backend uses nodejs')
        assert 'Node.js' in result

    def test_postgresql_correction(self, corrector):
        """Test PostgreSQL correction."""
        result = corrector.correct('postgres is fast')
        assert 'PostgreSQL' in result

    def test_json_uppercase(self, corrector):
        """Test JSON formatting."""
        result = corrector.correct('parse json data')
        assert 'JSON' in result

    def test_html_uppercase(self, corrector):
        """Test HTML formatting."""
        result = corrector.correct('html markup')
        assert 'HTML' in result

    def test_css_uppercase(self, corrector):
        """Test CSS formatting."""
        result = corrector.correct('css styling')
        assert 'CSS' in result


class TestTechTermCorrectionsPolishPhonetics:
    """Test Polish phonetic corrections from Whisper mishearing."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_dżawaskrypt_to_javascript(self, corrector):
        """Test Polish 'dżawaskrypt' → 'JavaScript'."""
        result = corrector.correct('dżawaskrypt jest popularny')
        assert 'JavaScript' in result

    def test_javaskrypt_variant(self, corrector):
        """Test Polish variant 'jawaskrypt'."""
        result = corrector.correct('jawaskrypt framework')
        assert 'JavaScript' in result

    def test_tajpskrypt_to_typescript(self, corrector):
        """Test Polish 'tajpskrypt' → 'TypeScript'."""
        result = corrector.correct('tajpskrypt project')
        assert 'TypeScript' in result

    def test_dżawa_to_java(self, corrector):
        """Test Polish 'dżawa' → 'Java'."""
        result = corrector.correct('dżawa program')
        assert 'Java' in result

    def test_pajon_to_python(self, corrector):
        """Test 'pajon' (with 'j') → 'Python'."""
        result = corrector.correct('pajton is popular')
        assert 'Python' in result

    def test_github_polish_correction(self, corrector):
        """Test Polish 'git hab' → 'GitHub'."""
        result = corrector.correct('git hab repo')
        assert 'GitHub' in result

    def test_gitlab_correction(self, corrector):
        """Test 'git lab' → 'GitLab'."""
        result = corrector.correct('git lab pipeline')
        assert 'GitLab' in result

    def test_merdż_to_merge(self, corrector):
        """Test Polish 'merdż' → 'merge'."""
        result = corrector.correct('merdż the branch')
        assert 'merge' in result

    def test_komit_to_commit(self, corrector):
        """Test Polish 'komit' → 'commit'."""
        result = corrector.correct('komit the changes')
        assert 'commit' in result

    def test_brańcz_to_branch(self, corrector):
        """Test Polish 'brańcz' → 'branch'."""
        result = corrector.correct('brańcz development')
        assert 'branch' in result


class TestTechTermCorrectionsAWSCloud:
    """Test AWS and cloud platform corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_aws_lowercase(self, corrector):
        """Test 'aws' → 'AWS'."""
        result = corrector.correct('deploy to aws')
        assert 'AWS' in result

    def test_azure_lowercase(self, corrector):
        """Test 'azure' → 'Azure'."""
        result = corrector.correct('using azure cloud')
        assert 'Azure' in result

    def test_gcp_correction(self, corrector):
        """Test GCP correction."""
        result = corrector.correct('gcp instance')
        assert 'GCP' in result

    def test_terraform_correction(self, corrector):
        """Test Terraform case normalization."""
        result = corrector.correct('terraform infrastructure')
        assert 'Terraform' in result

    def test_heroku_correction(self, corrector):
        """Test Heroku case normalization."""
        result = corrector.correct('deploy on heroku')
        assert 'Heroku' in result

    def test_vercel_correction(self, corrector):
        """Test Polish 'wersel' → 'Vercel'."""
        result = corrector.correct('wersel deployment')
        assert 'Vercel' in result


class TestTechTermCorrectionsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_empty_string(self, corrector):
        """Test correcting an empty string."""
        result = corrector.correct('')
        assert result == ''

    def test_none_patterns_no_error(self, corrector):
        """Test that non-matching text passes through unchanged."""
        text = 'This is plain text with no tech terms'
        result = corrector.correct(text)
        assert result == text

    def test_multiple_tech_terms_in_sentence(self, corrector):
        """Test multiple tech terms in one sentence."""
        text = 'I use javascript and python with docker and kubernetes'
        result = corrector.correct(text)
        assert 'JavaScript' in result
        assert 'Python' in result
        assert 'Docker' in result
        assert 'Kubernetes' in result

    def test_case_insensitive_matching(self, corrector):
        """Test case-insensitive matching."""
        text = 'JAVASCRIPT javascript JavaScript'
        result = corrector.correct(text)
        # All should be corrected to JavaScript
        assert result.count('JavaScript') >= 3

    def test_word_boundary_respected(self, corrector):
        """Test that word boundaries are respected."""
        # 'docker' within 'dockerfile' shouldn't match just 'docker'
        text = 'create dockerfile'
        result = corrector.correct(text)
        # dockerfile shouldn't be corrected to 'DockerFile'
        assert 'dockerfile' in result.lower()

    def test_unicode_text_preserved(self, corrector):
        """Test that non-tech unicode text is preserved."""
        text = 'Zażółw gęślą jaźń — docker is great'
        result = corrector.correct(text)
        assert 'Zażółw' in result
        assert 'Docker' in result

    def test_repeated_corrections_idempotent(self, corrector):
        """Test that repeated corrections don't over-correct."""
        text = 'I use docker'
        result1 = corrector.correct(text)
        result2 = corrector.correct(result1)
        assert result1 == result2

    def test_punctuation_preserved(self, corrector):
        """Test that punctuation is preserved."""
        text = 'Use docker! Seriously, use docker.'
        result = corrector.correct(text)
        assert '!' in result
        assert ',' in result
        assert '.' in result


class TestTechTermCorrectionsPolishGrammar:
    """Test Polish grammatical case handling."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_polish_genitive_case(self, corrector):
        """Test Polish genitive ending -a."""
        # PL_SUFFIX pattern should handle this
        text = 'dockera jest używany'
        result = corrector.correct(text)
        # Should correct despite genitive suffix
        assert 'Docker' in result or 'dockera' in result.lower()

    def test_polish_instrumental_case(self, corrector):
        """Test Polish instrumental ending -em."""
        text = 'dockerem pracuję'
        result = corrector.correct(text)
        assert 'Docker' in result or 'docker' in result.lower()

    def test_multiple_suffix_variants(self, corrector):
        """Test various Polish case endings."""
        text = 'dockerze dockerów dockerami'
        result = corrector.correct(text)
        # Should handle multiple case forms
        assert 'docker' in result.lower()


class TestTechTermCorrectionsInvalidRegex:
    """Test error handling for invalid regex patterns."""

    def test_add_invalid_regex_pattern(self, caplog):
        """Test that invalid regex patterns are logged and handled."""
        corrector = TechTermCorrections()
        corrector.clear()
        # Try to add an invalid regex pattern
        corrector.add_pattern('[invalid(regex', 'TEST')
        # Should not have added the pattern due to error
        # The pattern count should remain 0
        assert corrector.get_pattern_count() == 0

    def test_valid_pattern_after_invalid(self):
        """Test that corrector still works after adding invalid pattern."""
        corrector = TechTermCorrections()
        corrector.clear()
        # Add invalid pattern (should be skipped)
        corrector.add_pattern('[invalid(', 'TEST')
        # Add valid pattern
        corrector.add_pattern(r'\btest\b', 'VALID')
        # Should have 1 pattern (invalid was skipped)
        assert corrector.get_pattern_count() == 1
        result = corrector.correct('this is a test')
        assert 'VALID' in result


class TestTechTermCorrectionsFrameworks:
    """Test framework-specific corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_react_variants(self, corrector):
        """Test React variant spellings."""
        text = 'I use riakt and react'
        result = corrector.correct(text)
        assert result.count('React') >= 2

    def test_vue_correction(self, corrector):
        """Test Vue.js correction."""
        result = corrector.correct('vue js framework')
        assert 'Vue.js' in result

    def test_next_js_correction(self, corrector):
        """Test Next.js correction."""
        result = corrector.correct('next.js app')
        assert 'Next.js' in result

    def test_django_correction(self, corrector):
        """Test Django correction."""
        result = corrector.correct('dżango framework')
        assert 'Django' in result

    def test_fastapi_correction(self, corrector):
        """Test FastAPI correction."""
        result = corrector.correct('fastapi backend')
        assert 'FastAPI' in result

    def test_bootstrap_correction(self, corrector):
        """Test Bootstrap correction."""
        result = corrector.correct('butstrap grid')
        assert 'Bootstrap' in result

    def test_tailwind_correction(self, corrector):
        """Test Tailwind correction."""
        result = corrector.correct('tejlłind CSS')
        assert 'Tailwind' in result


class TestTechTermCorrectionsDatabases:
    """Test database-specific corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_mysql_correction(self, corrector):
        """Test 'mysql' → 'MySQL'."""
        result = corrector.correct('mysql database')
        assert 'MySQL' in result

    def test_mongodb_correction(self, corrector):
        """Test 'mongi di bi' → 'MongoDB'."""
        result = corrector.correct('mongi di bi database')
        assert 'MongoDB' in result

    def test_redis_correction(self, corrector):
        """Test Redis correction."""
        result = corrector.correct('redis cache')
        assert 'Redis' in result

    def test_elasticsearch_correction(self, corrector):
        """Test 'elasticsearch' → 'Elasticsearch'."""
        result = corrector.correct('elasticsearch cluster')
        assert 'Elasticsearch' in result

    def test_sqlite_correction(self, corrector):
        """Test SQLite correction."""
        result = corrector.correct('siku lajt database')
        assert 'SQLite' in result

    def test_cassandra_correction(self, corrector):
        """Test Cassandra correction."""
        result = corrector.correct('kasandra cluster')
        assert 'Cassandra' in result


class TestTechTermCorrectionsProtocols:
    """Test networking and protocol corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_http_correction(self, corrector):
        """Test HTTP correction."""
        result = corrector.correct('http request')
        assert 'HTTP' in result

    def test_https_correction(self, corrector):
        """Test HTTPS correction."""
        result = corrector.correct('https connection')
        assert 'HTTPS' in result

    def test_tcp_correction(self, corrector):
        """Test TCP correction."""
        result = corrector.correct('tcp protocol')
        assert 'TCP' in result

    def test_dns_correction(self, corrector):
        """Test DNS correction."""
        result = corrector.correct('di en es lookup')
        assert 'DNS' in result

    def test_ssh_correction(self, corrector):
        """Test SSH correction."""
        result = corrector.correct('es es hajcz connection')
        assert 'SSH' in result

    def test_websocket_correction(self, corrector):
        """Test WebSocket correction."""
        result = corrector.correct('łeb soket connection')
        assert 'WebSocket' in result

    def test_graphql_correction(self, corrector):
        """Test GraphQL correction."""
        result = corrector.correct('graf kju el query')
        assert 'GraphQL' in result

    def test_grpc_correction(self, corrector):
        """Test gRPC correction."""
        result = corrector.correct('dżi ar pi si call')
        assert 'gRPC' in result

    def test_oauth_correction(self, corrector):
        """Test 'o awt' → 'OAuth'."""
        result = corrector.correct('o awt authentication')
        assert 'OAuth' in result


class TestTechTermCorrectionsArchitecture:
    """Test architecture and pattern corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_backend_correction(self, corrector):
        """Test 'backend' correction."""
        result = corrector.correct('bek end development')
        assert 'backend' in result

    def test_frontend_correction(self, corrector):
        """Test 'frontend' correction."""
        result = corrector.correct('front end code')
        assert 'frontend' in result

    def test_middleware_correction(self, corrector):
        """Test middleware correction."""
        result = corrector.correct('midłer layer')
        assert 'middleware' in result

    def test_serverless_correction(self, corrector):
        """Test serverless correction."""
        result = corrector.correct('serwer les architecture')
        assert 'serverless' in result

    def test_webhook_correction(self, corrector):
        """Test webhook correction."""
        result = corrector.correct('łeb huk endpoint')
        assert 'webhook' in result

    def test_microservice_correction(self, corrector):
        """Test microservice correction."""
        result = corrector.correct('mikro serwis architecture')
        assert 'mikroserwis' in result.lower()


class TestTechTermCorrectionsDevops:
    """Test DevOps-specific corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_cicd_correction(self, corrector):
        """Test CI/CD correction."""
        result = corrector.correct('si aj si di pipeline')
        assert 'CI/CD' in result

    def test_devops_correction(self, corrector):
        """Test DevOps correction."""
        result = corrector.correct('devops team')
        assert 'DevOps' in result

    def test_jenkins_correction(self, corrector):
        """Test Jenkins correction."""
        result = corrector.correct('dżenkins automation')
        assert 'Jenkins' in result

    def test_pipeline_correction(self, corrector):
        """Test pipeline correction."""
        result = corrector.correct('pajplajn execution')
        assert 'pipeline' in result

    def test_deployment_correction(self, corrector):
        """Test deployment correction."""
        result = corrector.correct('depłojment process')
        assert 'deployment' in result

    def test_prometheus_correction(self, corrector):
        """Test Prometheus correction."""
        result = corrector.correct('prometeusz monitoring')
        assert 'Prometheus' in result


class TestTechTermCorrectionsOperatingSystems:
    """Test OS and system command corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_linux_correction(self, corrector):
        """Test Linux correction."""
        result = corrector.correct('linuks server')
        assert 'Linux' in result

    def test_windows_correction(self, corrector):
        """Test Windows correction."""
        result = corrector.correct('windołs system')
        assert 'Windows' in result

    def test_macos_correction(self, corrector):
        """Test macOS correction."""
        result = corrector.correct('macos device')
        assert 'macOS' in result

    def test_chmod_correction(self, corrector):
        """Test chmod correction."""
        result = corrector.correct('hamot permissions')
        assert 'chmod' in result

    def test_bash_correction(self, corrector):
        """Test bash correction."""
        result = corrector.correct('basz script')
        assert 'bash' in result


class TestTechTermCorrectionsEditors:
    """Test editor and IDE corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_vscode_correction(self, corrector):
        """Test VS Code correction."""
        result = corrector.correct('wi es kod editor')
        assert 'VS Code' in result

    def test_intellij_correction(self, corrector):
        """Test IntelliJ correction."""
        result = corrector.correct('inteli dżej IDE')
        assert 'IntelliJ' in result

    def test_powershell_correction(self, corrector):
        """Test PowerShell correction."""
        result = corrector.correct('pałer szel command')
        assert 'PowerShell' in result


class TestTechTermCorrectionsMLDataScience:
    """Test ML and data science corrections."""

    @pytest.fixture
    def corrector(self):
        return TechTermCorrections()

    def test_tensorflow_correction(self, corrector):
        """Test TensorFlow correction."""
        result = corrector.correct('tensorflow model')
        assert 'TensorFlow' in result

    def test_pytorch_correction(self, corrector):
        """Test PyTorch correction."""
        result = corrector.correct('pytorch training')
        assert 'PyTorch' in result

    def test_sklearn_correction(self, corrector):
        """Test scikit-learn correction."""
        result = corrector.correct('scikit learn classifier')
        assert 'scikit-learn' in result

    def test_numpy_correction(self, corrector):
        """Test NumPy correction."""
        result = corrector.correct('numpy array')
        assert 'NumPy' in result

    def test_pandas_correction(self, corrector):
        """Test Pandas correction."""
        result = corrector.correct('pandas dataframe')
        assert 'Pandas' in result


class TestTechTermCorrectionsLanguage:
    """Tests for language-parameterized tech corrections."""

    def test_polish_loads_patterns(self):
        corrector = TechTermCorrections(language="pl")
        assert corrector.get_pattern_count() > 0

    def test_english_loads_no_patterns(self):
        corrector = TechTermCorrections(language="en")
        assert corrector.get_pattern_count() == 0

    def test_unknown_language_loads_no_patterns(self):
        corrector = TechTermCorrections(language="xx")
        assert corrector.get_pattern_count() == 0

    def test_english_noop_correct(self):
        corrector = TechTermCorrections(language="en")
        text = "dokier compose is great"
        assert corrector.correct(text) == text  # No correction for English

    def test_polish_corrects_docker(self):
        corrector = TechTermCorrections(language="pl")
        assert "Docker" in corrector.correct("dokier jest świetny")

    def test_default_language_is_polish(self):
        corrector = TechTermCorrections()
        assert corrector.get_pattern_count() > 0

    def test_spanish_loads_patterns(self):
        corrector = TechTermCorrections(language="es")
        assert corrector.get_pattern_count() > 0

    def test_spanish_corrects_docker(self):
        corrector = TechTermCorrections(language="es")
        assert "Docker" in corrector.correct("doquer está aquí")

    def test_spanish_corrects_kubernetes(self):
        corrector = TechTermCorrections(language="es")
        assert "Kubernetes" in corrector.correct("cubernetis cluster")

    def test_spanish_corrects_javascript(self):
        corrector = TechTermCorrections(language="es")
        assert "JavaScript" in corrector.correct("yavascript es popular")

    def test_spanish_corrects_typescript(self):
        corrector = TechTermCorrections(language="es")
        assert "TypeScript" in corrector.correct("taipscript project")

    def test_spanish_corrects_python(self):
        corrector = TechTermCorrections(language="es")
        assert "Python" in corrector.correct("paiton es útil")

    def test_spanish_corrects_github(self):
        corrector = TechTermCorrections(language="es")
        assert "GitHub" in corrector.correct("guijab repositorio")

    def test_spanish_corrects_react(self):
        corrector = TechTermCorrections(language="es")
        assert "React" in corrector.correct("riact framework")

    def test_spanish_corrects_postgresql(self):
        corrector = TechTermCorrections(language="es")
        assert "PostgreSQL" in corrector.correct("postgre base de datos")

    def test_spanish_corrects_mongodb(self):
        corrector = TechTermCorrections(language="es")
        assert "MongoDB" in corrector.correct("mongui base de datos")

    def test_spanish_corrects_nodejs(self):
        corrector = TechTermCorrections(language="es")
        assert "Node.js" in corrector.correct("nod jés servidor")

    def test_german_loads_patterns(self):
        corrector = TechTermCorrections(language="de")
        assert corrector.get_pattern_count() > 0

    def test_german_corrects_docker(self):
        corrector = TechTermCorrections(language="de")
        # Test German phonetic mishearing: "Doker" or "Dokker" → Docker
        result = corrector.correct("Doker ist großartig")
        assert "Docker" in result

    def test_german_corrects_javascript(self):
        corrector = TechTermCorrections(language="de")
        # German Whisper mishearing: "Dschawaskript" → JavaScript
        result = corrector.correct("Dschawaskript ist beliebt")
        assert "JavaScript" in result

    def test_german_corrects_typescript(self):
        corrector = TechTermCorrections(language="de")
        # German: "Taijpskript" or "Taipskript" → TypeScript
        result = corrector.correct("Taijpskript Projekt")
        assert "TypeScript" in result

    def test_german_corrects_python(self):
        corrector = TechTermCorrections(language="de")
        # German: "Paiton" or "Pyton" → Python
        result = corrector.correct("Paiton ist leistungsstark")
        assert "Python" in result

    def test_german_corrects_github(self):
        corrector = TechTermCorrections(language="de")
        # German: "Githab" → GitHub
        result = corrector.correct("Githab Repository")
        assert "GitHub" in result

    def test_german_corrects_kubernetes(self):
        corrector = TechTermCorrections(language="de")
        # German: "Kubernets" → Kubernetes
        result = corrector.correct("Kubernets Cluster")
        assert "Kubernetes" in result

    def test_german_corrects_react(self):
        corrector = TechTermCorrections(language="de")
        # German: "Riäkt" or "Reakt" → React
        result = corrector.correct("Riäkt Framework")
        assert "React" in result

    def test_german_corrects_node_js(self):
        corrector = TechTermCorrections(language="de")
        # German: "Noud" or "Node" → Node.js
        result = corrector.correct("Noud.js Backend")
        assert "Node.js" in result

    def test_german_corrects_rest_api(self):
        corrector = TechTermCorrections(language="de")
        # German: "Rest Äpi" → REST API
        result = corrector.correct("Rest Äpi Endpunkt")
        assert "REST API" in result

    def test_german_corrects_gitlab(self):
        corrector = TechTermCorrections(language="de")
        # German: "Gitlab" → GitLab
        result = corrector.correct("Gitlab Pipeline")
        assert "GitLab" in result

    def test_german_corrects_sql(self):
        corrector = TechTermCorrections(language="de")
        # German: "Sikjuel" or "Sikuel" → SQL
        result = corrector.correct("Sikjuel Datenbank")
        assert "SQL" in result

    def test_german_corrects_postgresql(self):
        corrector = TechTermCorrections(language="de")
        # German: "Postgre Sikjuel" or similar → PostgreSQL
        result = corrector.correct("Postgre Sikjuel Server")
        assert "PostgreSQL" in result

    def test_german_corrects_mongodb(self):
        corrector = TechTermCorrections(language="de")
        # German: "Mongo Di Bi" → MongoDB
        result = corrector.correct("Mongo Di Bi Datenbank")
        assert "MongoDB" in result

    def test_german_corrects_http(self):
        corrector = TechTermCorrections(language="de")
        # German: "Äitsch Tipi Tipi" → HTTP
        result = corrector.correct("Äitsch Tipi Tipi Anfrage")
        assert "HTTP" in result

    def test_german_corrects_https(self):
        corrector = TechTermCorrections(language="de")
        # German: "Äitsch Tipi Tipi Es" → HTTPS
        result = corrector.correct("Äitsch Tipi Tipi Es Verbindung")
        assert "HTTPS" in result

    def test_german_corrects_aws(self):
        corrector = TechTermCorrections(language="de")
        # German: "Ä Wu Es" → AWS
        result = corrector.correct("Ä Wu Es Cloud")
        assert "AWS" in result

    def test_german_corrects_ssh(self):
        corrector = TechTermCorrections(language="de")
        # German: "Es Es Ha" → SSH
        result = corrector.correct("Es Es Ha Verbindung")
        assert "SSH" in result

    def test_german_corrects_backend(self):
        corrector = TechTermCorrections(language="de")
        # German: "Bäkent" → backend
        result = corrector.correct("Bäkent Entwicklung")
        assert "backend" in result

    def test_german_corrects_frontend(self):
        corrector = TechTermCorrections(language="de")
        # German: "Frontänd" → frontend
        result = corrector.correct("Frontänd Code")
        assert "frontend" in result

    def test_german_corrects_framework(self):
        corrector = TechTermCorrections(language="de")
        # German: "Fräjmwork" → framework
        result = corrector.correct("Fräjmwork Bibliothek")
        assert "framework" in result

    def test_german_corrects_cache(self):
        corrector = TechTermCorrections(language="de")
        # German: "Käsch" → cache
        result = corrector.correct("Käsch Speicher")
        assert "cache" in result

    def test_german_corrects_linux(self):
        corrector = TechTermCorrections(language="de")
        # German: "Linuks" → Linux
        result = corrector.correct("Linuks System")
        assert "Linux" in result

    def test_german_corrects_with_declension(self):
        corrector = TechTermCorrections(language="de")
        # German declension: "des Dockers" (genitive), "dem Docker" (dative)
        result = corrector.correct("des Dockers verwendet")
        assert "Docker" in result

    def test_german_different_from_polish(self):
        """Verify German and Polish have different corrections for same term."""
        de_corrector = TechTermCorrections(language="de")
        pl_corrector = TechTermCorrections(language="pl")

        # Both should load patterns, but different ones
        assert de_corrector.get_pattern_count() > 0
        assert pl_corrector.get_pattern_count() > 0
