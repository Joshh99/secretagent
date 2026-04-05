"""Direct Python implementations for mid-level MedAgentBench ptools.

These wrap fhir_tools with domain logic so orchestrate-generated
workflows can call high-level operations instead of constructing
raw FHIR URLs and parsing JSON bundles.
"""

import json
from datetime import datetime, timedelta

import fhir_tools


def find_patient_impl(fhir_base, family, given, birthdate):
    url = f"{fhir_base}Patient?family={family}&given={given}&birthdate={birthdate}"
    response = fhir_tools.fhir_get(url)
    try:
        data = json.loads(response)
        if 'entry' in data and len(data['entry']) > 0:
            for ident in data['entry'][0]['resource'].get('identifier', []):
                return ident.get('value', 'Patient not found')
        return 'Patient not found'
    except (json.JSONDecodeError, KeyError, IndexError):
        return 'Patient not found'


def get_patient_dob_impl(fhir_base, mrn):
    url = f"{fhir_base}Patient?identifier={mrn}"
    response = fhir_tools.fhir_get(url)
    try:
        data = json.loads(response)
        return data['entry'][0]['resource']['birthDate']
    except (json.JSONDecodeError, KeyError, IndexError):
        return ''


def get_observations_impl(fhir_base, mrn, code, hours):
    url = f"{fhir_base}Observation?patient={mrn}&code={code}&_count=5000"
    response = fhir_tools.fhir_get(url)
    try:
        data = json.loads(response)
        if 'entry' not in data:
            return '[]'

        cutoff = None
        if hours > 0:
            cutoff = datetime.fromisoformat('2023-11-13T10:15:00+00:00') - timedelta(hours=hours)

        results = []
        for entry in data['entry']:
            resource = entry['resource']
            effective = datetime.fromisoformat(resource['effectiveDateTime'])
            if cutoff and effective < cutoff:
                continue
            value = resource.get('valueQuantity', {}).get('value')
            if value is not None:
                results.append({'value': value, 'time': effective.isoformat()})

        results.sort(key=lambda x: x['time'], reverse=True)
        return json.dumps(results)
    except (json.JSONDecodeError, KeyError, ValueError):
        return '[]'


def calculate_age_impl(dob, reference_date):
    dob_dt = datetime.fromisoformat(dob[:10])
    ref_dt = datetime.fromisoformat(reference_date[:10])
    age = ref_dt.year - dob_dt.year
    if (ref_dt.month, ref_dt.day) < (dob_dt.month, dob_dt.day):
        age -= 1
    return age


def record_vital_impl(fhir_base, mrn, code, value, timestamp):
    payload = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category",
                                   "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"text": code},
        "subject": {"reference": f"Patient/{mrn}"},
        "effectiveDateTime": timestamp,
        "valueString": value,
    }
    return fhir_tools.fhir_post(f"{fhir_base}Observation", json.dumps(payload))


def create_order_impl(fhir_base, mrn, order_type, params, timestamp):
    p = json.loads(params) if isinstance(params, str) else params

    if order_type == 'medication':
        payload = {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://hl7.org/fhir/sid/ndc",
                            "code": p['ndc'], "display": p.get('display', '')}],
                "text": p.get('display', ''),
            },
            "subject": {"reference": f"Patient/{mrn}"},
            "authoredOn": timestamp,
            "dosageInstruction": [{
                "route": p.get('route', 'IV'),
                "doseAndRate": [{
                    "doseQuantity": {"value": p['dose'], "unit": p.get('dose_unit', 'g')},
                    "rateQuantity": {"value": p['rate'], "unit": p.get('rate_unit', 'h')},
                }],
            }],
        }
        return fhir_tools.fhir_post(f"{fhir_base}MedicationRequest", json.dumps(payload))

    elif order_type == 'service':
        payload = {
            "resourceType": "ServiceRequest",
            "status": "active",
            "intent": "order",
            "priority": p.get('priority', 'stat'),
            "code": {"coding": [{"system": p['system'], "code": p['code'],
                                  "display": p.get('display', '')}]},
            "subject": {"reference": f"Patient/{mrn}"},
            "authoredOn": timestamp,
        }
        if p.get('note'):
            payload["note"] = {"text": p['note']}
        if p.get('occurrence'):
            payload["occurrenceDateTime"] = p['occurrence']
        return fhir_tools.fhir_post(f"{fhir_base}ServiceRequest", json.dumps(payload))

    return f"Unknown order type: {order_type}"
