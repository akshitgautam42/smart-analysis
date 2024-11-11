import pytest
import pandas as pd

@pytest.fixture
def sample_flight_data():
    """Fixture providing sample flight data for testing"""
    return pd.DataFrame({
        'PREFIXcustomer_id': ['PRE123', 'PRE456', 'PRE789'],
        'FLIGHT_NUM': ['FL100', 'FL200', 'FL300'],
        'departure_TIME': ['09:00', '10:00', '11:00'],
        'price_USD': ['$100', '$200', '$300'],
        'is_cancelled': ['Yes', 'No', 'No']
    })

@pytest.fixture
def sample_airline_data():
    """Fixture providing sample airline mapping data for testing"""
    return pd.DataFrame({
        'flight_id': [1, 2, 3],
        'airline': ['AA', 'UA', 'DL'],
        'destination': ['JFK', 'LAX', 'SFO']
    })

@pytest.fixture
def mock_openai_response():
    """Fixture providing mock OpenAI API response"""
    return {
        "choices": [{
            "message": {
                "content": '{"cleaning_operations": []}'
            }
        }]
    }