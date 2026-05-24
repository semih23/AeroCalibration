#!/usr/bin/env python3
"""Downloads METAR data from Iowa Environmental Mesonet (IEM) archive."""

import urllib.request
import urllib.parse
import time
import sys
import json
from datetime import datetime
from pathlib import Path

from config import AIRPORTS, RAW_DIR, TRAIN_YEARS, TEST_YEARS

START_YEAR = TRAIN_YEARS[0]
END_YEAR = TEST_YEARS[1]

def download_metar(icao: str, start_year: int, end_year: int, 
                   max_retries: int = 3) -> tuple:
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    
    params = {
        'station': icao,
        'data': 'all',
        'year1': str(start_year),
        'month1': '1',
        'day1': '1',
        'year2': str(end_year),
        'month2': '12',
        'day2': '31',
        'tz': 'UTC',
        'format': 'comma',
        'latlon': 'no',
        'elev': 'no',
        'missing': 'M',
        'trace': 'T',
        'direct': 'no',
        'report_type': '3'
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'AeroCalib Research Project (Academic)')
            
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read().decode('utf-8')
                
                lines = data.strip().split('\n')
                data_lines = [l for l in lines if not l.startswith('#')]
                n_records = len(data_lines) - 1
                
                if n_records < 100:
                    return (icao, False, 0, f"Too few records: {n_records}")
                
                filename = RAW_DIR / f"metar_{icao}.csv"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(data)
                
                return (icao, True, n_records, None)
                
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return (icao, False, 0, str(e))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return (icao, False, 0, str(e))
    
    return (icao, False, 0, "Max retries exceeded")

def main():
    test_mode = '--test' in sys.argv
    if test_mode:
        test_airports = ['KJFK', 'EGLL', 'WSSS']
        airports_to_download = {k: v for k, v in AIRPORTS.items() if k in test_airports}
        print(f"TEST MODE: Downloading {len(airports_to_download)} airports only")
    else:
        airports_to_download = AIRPORTS
    
    print(f"Airports: {len(airports_to_download)}")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Output: {RAW_DIR}/")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        'success': [],
        'failed': [],
        'total_records': 0,
        'start_time': datetime.now().isoformat(),
        'period': f"{START_YEAR}-{END_YEAR}"
    }
    
    total = len(airports_to_download)
    for idx, (icao, info) in enumerate(airports_to_download.items(), 1):
        print(f"\n[{idx}/{total}] {icao} - {info['name']} ({info['city']}, {info['country']})")
        
        icao_result, success, n_records, error = download_metar(icao, START_YEAR, END_YEAR)
        
        if success:
            print(f"  OK: {n_records:,} records")
            results['success'].append({
                'icao': icao,
                'name': info['name'],
                'records': n_records
            })
            results['total_records'] += n_records
        else:
            print(f"  FAILED: {error}")
            results['failed'].append({
                'icao': icao,
                'error': error
            })
        
        if idx < total:
            time.sleep(2)
    
    results['end_time'] = datetime.now().isoformat()
    results['n_success'] = len(results['success'])
    results['n_failed'] = len(results['failed'])
    
    log_file = RAW_DIR / 'download_log.json'
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print(f"Successful: {len(results['success'])}/{total} airports")
    print(f"Failed: {len(results['failed'])}/{total} airports")
    print(f"Total records: {results['total_records']:,}")
    print(f"Log saved: {log_file}")
    
    if results['failed']:
        print("\nFailed airports:")
        for item in results['failed']:
            print(f"  - {item['icao']}: {item['error']}")
    
    return len(results['failed']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
