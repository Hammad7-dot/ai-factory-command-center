import io
import json
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from factory.llm import generate_explanation

INCIDENT={'recommendation':{'action':'maintenance','reason':'Review required.'},'evidence':[]}

def test_missing_key_does_not_call_provider():
    with patch.dict('os.environ',{'OPENAI_API_KEY':''}),patch('urllib.request.urlopen') as call:
        assert generate_explanation(INCIDENT)['error_code']=='missing_key'
        call.assert_not_called()

def test_http_diagnostics_never_echo_key():
    for status,code,expected in [(401,'invalid_api_key','authentication_failed'),(429,'insufficient_quota','insufficient_quota'),(429,None,'unclassified_429'),(404,'model_not_found','model_not_found'),(500,None,'provider_unavailable')]:
        body=json.dumps({'error':{'code':code,'message':'bad key sk-secret-for-test'}}).encode()
        error=HTTPError('https://api.openai.com',status,'failure',{},io.BytesIO(body))
        with patch('urllib.request.urlopen',side_effect=error):
            result=generate_explanation(INCIDENT,'sk-secret-for-test')
        assert result['error_code']==expected
        assert 'sk-secret-for-test' not in json.dumps(result)
        assert not result['available']

def test_specific_quota_types_and_retry_after():
    for code in ['credit_balance_exhausted','organization_spend_limit_exceeded','project_spend_limit_exceeded','organization_usage_limit_exceeded']:
        error=HTTPError('https://api.openai.com',429,'failure',{},io.BytesIO(json.dumps({'error':{'code':code,'type':'insufficient_quota'}}).encode()))
        with patch('urllib.request.urlopen',side_effect=error):
            assert generate_explanation(INCIDENT,'test-key')['error_code']==code
    error=HTTPError('https://api.openai.com',429,'failure',{'Retry-After':'12'},io.BytesIO(b'{"error":{"code":"rate_limit_exceeded"}}'))
    with patch('urllib.request.urlopen',side_effect=error):
        result=generate_explanation(INCIDENT,'test-key')
        assert result['error_code']=='rate_limit' and '12 seconds' in result['reason']

def test_network_and_success():
    with patch('urllib.request.urlopen',side_effect=URLError('blocked')):
        assert generate_explanation(INCIDENT,'test-key')['error_code']=='network_error'
    response=io.BytesIO(json.dumps({'choices':[{'message':{'content':'Supervisor review required.'}}]}).encode())
    with patch('urllib.request.urlopen',return_value=response) as call:
        result=generate_explanation(INCIDENT,'  test-key  ')
        assert call.call_args.args[0].get_header('Authorization')=='Bearer test-key'
        assert result['mode']=='hosted_llm'
        assert result['available']
