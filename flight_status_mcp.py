import json
import os  
import requests
import pytz
from datetime import datetime, timedelta
from typing import Any, Callable, Set, Dict, List, Optional
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Flight Server")

@mcp.tool()
def get_flight_status(flight):
    """Returns Flight Information"""
    AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
    AEROAPI_KEY="xxxx"
 
    def get_api_session():
        session = requests.Session()
        session.headers.update({"x-apikey": AEROAPI_KEY})
        return session
 
    def fetch_flight_data(flight_id, session):
        if "flight_id=" in flight_id:
            flight_id = flight_id.split("flight_id=")[1]    
        
        start_date = datetime.now().date().strftime('%Y-%m-%d')
        end_date = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
        api_resource = f"/flights/{flight_id}?start={start_date}&amp;end={end_date}"  # Fixed &amp; to &amp;
        
        try:
            response = session.get(f"{AEROAPI_BASE_URL}{api_resource}")
            response.raise_for_status()
            json_data = response.json()
            
            # Print the response for debugging
            print("API Response:", json.dumps(json_data, indent=2))
            
            if 'flights' not in json_data or len(json_data['flights']) == 0:
                return None
            
            return json_data['flights'][0]
        except Exception as e:
            print(f"Error fetching flight data: {e}")
            return None
 
    def utc_to_local(utc_date_str, local_timezone_str):
        utc_datetime = datetime.strptime(utc_date_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.utc)
        local_timezone = pytz.timezone(local_timezone_str)
        local_datetime = utc_datetime.astimezone(local_timezone)
        return local_datetime.strftime('%Y-%m-%d %H:%M:%S')    
    
    session = get_api_session()
    flight_data = fetch_flight_data(flight, session)
    
    if flight_data is None:
        return json.dumps({
            'error': 'Failed to retrieve flight data. Check the API key and flight ID.',
            'flight': flight
        })
    
    try:
        dep_key = 'estimated_out' if 'estimated_out' in flight_data and flight_data['estimated_out'] else \
              'actual_out' if 'actual_out' in flight_data and flight_data['actual_out'] else \
              'scheduled_out'
        
        arr_key = 'estimated_in' if 'estimated_in' in flight_data and flight_data['estimated_in'] else \
              'actual_in' if 'actual_in' in flight_data and flight_data['actual_in'] else \
              'scheduled_in'    
        
        flight_details = f"""
                        Flight: {flight}
                        Source: {flight_data['origin']['city']}
                        Destination: {flight_data['destination']['city']}
                        Departure Time: {utc_to_local(flight_data[dep_key], flight_data['origin']['timezone'])}
                        Arrival Time: {utc_to_local(flight_data[arr_key], flight_data['destination']['timezone'])}
                        Status: {flight_data['status']}
                        """
        return flight_details
    except Exception as e:
        print(f"Error processing flight data: {e}")
        return json.dumps({
            'error': f'Error processing flight data: {str(e)}',
            'flight': flight
        })
    
if __name__ == "__main__":  
    mcp.run()
