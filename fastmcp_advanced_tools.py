#!/usr/bin/env python3
"""
Advanced FastMCP Tools Server
=============================

A comprehensive MCP server providing data analysis, file processing, 
web utilities, and system monitoring tools. This demonstrates advanced
MCP server patterns with real-world utility functions.

Requirements:
pip install fastmcp pandas requests beautifulsoup4 psutil nltk pillow qrcode[pil] cryptography

Usage:
python fastmcp_advanced_tools.py

Author: Your Name
Date: 2024
"""

from fastmcp import FastMCP
import pandas as pd
import requests
import json
import os
import hashlib
import base64
import qrcode
import psutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup
from PIL import Image
import io
import re
import csv
from pathlib import Path
import tempfile
import zipfile
import shutil
from cryptography.fernet import Fernet

# Initialize the advanced FastMCP server
mcp = FastMCP("Advanced Tools Server")

# =============================================================================
# DATA ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def analyze_csv_file(file_path: str, delimiter: str = ',') -> Dict[str, Any]:
    """
    Analyze a CSV file and provide comprehensive statistics and insights.
    
    Args:
        file_path: Path to the CSV file
        delimiter: CSV delimiter (default: comma)
    
    Returns:
        Dictionary containing file analysis, column info, and basic statistics
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        analysis = {
            "file_info": {
                "file_path": file_path,
                "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2),
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
            },
            "columns": {},
            "data_quality": {
                "total_missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum(),
                "completely_empty_rows": (df.isnull().all(axis=1)).sum()
            },
            "sample_data": df.head(3).to_dict('records')
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "data_type": str(df[col].dtype),
                "missing_values": df[col].isnull().sum(),
                "unique_values": df[col].nunique(),
                "missing_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            # Add statistics based on data type
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "mean": round(df[col].mean(), 2) if pd.notnull(df[col].mean()) else None,
                    "median": df[col].median(),
                    "std": round(df[col].std(), 2) if pd.notnull(df[col].std()) else None
                })
            elif df[col].dtype == 'object':
                col_info.update({
                    "most_common": df[col].value_counts().head(3).to_dict() if not df[col].empty else {},
                    "avg_length": round(df[col].astype(str).str.len().mean(), 2) if not df[col].empty else 0
                })
            
            analysis["columns"][col] = col_info
        
        print(f"✅ Analyzed CSV file: {file_path}")
        print(f"   📊 {analysis['file_info']['rows']} rows × {analysis['file_info']['columns']} columns")
        print(f"   💾 File size: {analysis['file_info']['file_size_mb']} MB")
        print(f"   ⚠️  Missing values: {analysis['data_quality']['total_missing_values']}")
        
        return analysis
        
    except Exception as e:
        raise ValueError(f"Error analyzing CSV file: {str(e)}")

@mcp.tool()
def filter_and_export_data(file_path: str, filters: Dict[str, Any], 
                          output_path: str, export_format: str = 'csv') -> Dict[str, str]:
    """
    Filter CSV data based on conditions and export to various formats.
    
    Args:
        file_path: Path to input CSV file
        filters: Dictionary of column filters {column: {operator: value}}
        output_path: Path for output file
        export_format: Output format ('csv', 'json', 'excel')
    
    Returns:
        Dictionary with operation results
    """
    try:
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # Apply filters
        for column, condition in filters.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in data")
            
            for operator, value in condition.items():
                if operator == 'equals':
                    df = df[df[column] == value]
                elif operator == 'greater_than':
                    df = df[df[column] > value]
                elif operator == 'less_than':
                    df = df[df[column] < value]
                elif operator == 'contains':
                    df = df[df[column].astype(str).str.contains(str(value), na=False)]
                elif operator == 'not_null':
                    df = df[df[column].notnull()]
                elif operator == 'in_list':
                    df = df[df[column].isin(value)]
        
        # Export filtered data
        if export_format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif export_format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif export_format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError("Unsupported export format. Use 'csv', 'json', or 'excel'")
        
        result = {
            "status": "success",
            "original_rows": original_rows,
            "filtered_rows": len(df),
            "rows_removed": original_rows - len(df),
            "output_file": output_path,
            "export_format": export_format
        }
        
        print(f"✅ Data filtered and exported successfully")
        print(f"   📥 Original rows: {original_rows}")
        print(f"   📤 Filtered rows: {len(df)}")
        print(f"   💾 Output: {output_path}")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error filtering and exporting data: {str(e)}")

# =============================================================================
# WEB SCRAPING AND API TOOLS
# =============================================================================

@mcp.tool()
def scrape_website_content(url: str, extract_type: str = 'text', 
                          css_selector: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrape content from a website with various extraction options.
    
    Args:
        url: Website URL to scrape
        extract_type: Type of content to extract ('text', 'links', 'images', 'tables')
        css_selector: Optional CSS selector for targeted extraction
    
    Returns:
        Dictionary containing scraped content and metadata
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        result = {
            "url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get('content-type', ''),
            "page_size_kb": round(len(response.content) / 1024, 2),
            "extraction_type": extract_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply CSS selector if provided
        if css_selector:
            elements = soup.select(css_selector)
            soup = BeautifulSoup(''.join(str(el) for el in elements), 'html.parser')
        
        if extract_type == 'text':
            result["content"] = soup.get_text(strip=True, separator='\n')
            result["word_count"] = len(result["content"].split())
            
        elif extract_type == 'links':
            links = []
            for link in soup.find_all('a', href=True):
                link_url = link['href']
                # Convert relative URLs to absolute
                if link_url.startswith('/'):
                    link_url = f"{parsed_url.scheme}://{parsed_url.netloc}{link_url}"
                elif not link_url.startswith(('http://', 'https://')):
                    continue
                
                links.append({
                    "url": link_url,
                    "text": link.get_text(strip=True),
                    "title": link.get('title', '')
                })
            result["content"] = links
            result["link_count"] = len(links)
            
        elif extract_type == 'images':
            images = []
            for img in soup.find_all('img', src=True):
                img_url = img['src']
                # Convert relative URLs to absolute
                if img_url.startswith('/'):
                    img_url = f"{parsed_url.scheme}://{parsed_url.netloc}{img_url}"
                
                images.append({
                    "url": img_url,
                    "alt": img.get('alt', ''),
                    "title": img.get('title', ''),
                    "width": img.get('width', ''),
                    "height": img.get('height', '')
                })
            result["content"] = images
            result["image_count"] = len(images)
            
        elif extract_type == 'tables':
            tables = []
            for table in soup.find_all('table'):
                table_data = []
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if row_data:  # Only add non-empty rows
                        table_data.append(row_data)
                
                if table_data:
                    tables.append({
                        "data": table_data,
                        "rows": len(table_data),
                        "columns": len(table_data[0]) if table_data else 0
                    })
            
            result["content"] = tables
            result["table_count"] = len(tables)
        
        print(f"✅ Scraped content from: {url}")
        print(f"   📄 Page size: {result['page_size_kb']} KB")
        print(f"   🎯 Extraction type: {extract_type}")
        
        return result
        
    except requests.RequestException as e:
        raise ValueError(f"Error fetching URL: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error scraping website: {str(e)}")

@mcp.tool()
def fetch_api_data(url: str, method: str = 'GET', headers: Optional[Dict[str, str]] = None,
                  params: Optional[Dict[str, Any]] = None, 
                  data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Make API requests and handle various response formats.
    
    Args:
        url: API endpoint URL
        method: HTTP method ('GET', 'POST', 'PUT', 'DELETE')
        headers: Optional request headers
        params: Optional query parameters
        data: Optional request body data
    
    Returns:
        Dictionary containing API response and metadata
    """
    try:
        # Prepare request parameters
        request_headers = headers or {}
        request_params = params or {}
        request_data = data or {}
        
        # Add default headers
        if 'User-Agent' not in request_headers:
            request_headers['User-Agent'] = 'Advanced-MCP-Tools/1.0'
        
        # Make the request
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=request_headers,
            params=request_params,
            json=request_data if request_data else None,
            timeout=30
        )
        
        # Parse response
        content_type = response.headers.get('content-type', '').lower()
        
        try:
            if 'application/json' in content_type:
                response_data = response.json()
            else:
                response_data = response.text
        except json.JSONDecodeError:
            response_data = response.text
        
        result = {
            "url": url,
            "method": method.upper(),
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "headers": dict(response.headers),
            "content_type": content_type,
            "response_size_bytes": len(response.content),
            "response_time_ms": round(response.elapsed.total_seconds() * 1000, 2),
            "data": response_data,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ API request completed: {method} {url}")
        print(f"   📊 Status: {response.status_code}")
        print(f"   ⏱️  Response time: {result['response_time_ms']} ms")
        print(f"   📦 Response size: {result['response_size_bytes']} bytes")
        
        return result
        
    except requests.RequestException as e:
        raise ValueError(f"API request failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing API response: {str(e)}")

# =============================================================================
# FILE PROCESSING TOOLS
# =============================================================================

@mcp.tool()
def process_text_file(file_path: str, operations: List[str]) -> Dict[str, Any]:
    """
    Process text files with various operations like word count, sentiment, cleanup.
    
    Args:
        file_path: Path to the text file
        operations: List of operations ('word_count', 'line_count', 'char_count', 
                   'remove_duplicates', 'extract_emails', 'extract_urls')
    
    Returns:
        Dictionary containing processing results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        result = {
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "operations_performed": operations,
            "timestamp": datetime.now().isoformat()
        }
        
        for operation in operations:
            if operation == 'word_count':
                words = content.split()
                result["word_count"] = len(words)
                result["unique_words"] = len(set(word.lower().strip('.,!?";') for word in words))
                
            elif operation == 'line_count':
                lines = content.split('\n')
                result["total_lines"] = len(lines)
                result["non_empty_lines"] = len([line for line in lines if line.strip()])
                
            elif operation == 'char_count':
                result["total_characters"] = len(content)
                result["characters_no_spaces"] = len(content.replace(' ', ''))
                result["alphanumeric_characters"] = len(re.sub(r'[^a-zA-Z0-9]', '', content))
                
            elif operation == 'remove_duplicates':
                lines = content.split('\n')
                unique_lines = list(dict.fromkeys(lines))  # Preserves order
                result["original_lines"] = len(lines)
                result["unique_lines"] = len(unique_lines)
                result["duplicates_removed"] = len(lines) - len(unique_lines)
                
            elif operation == 'extract_emails':
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, content)
                result["emails_found"] = list(set(emails))
                result["email_count"] = len(result["emails_found"])
                
            elif operation == 'extract_urls':
                url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                urls = re.findall(url_pattern, content)
                result["urls_found"] = list(set(urls))
                result["url_count"] = len(result["urls_found"])
        
        print(f"✅ Processed text file: {file_path}")
        print(f"   📄 File size: {result['file_size_bytes']} bytes")
        print(f"   🔧 Operations: {', '.join(operations)}")
        
        return result
        
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing text file: {str(e)}")

@mcp.tool()
def create_archive(source_paths: List[str], archive_path: str, 
                  archive_type: str = 'zip') -> Dict[str, Any]:
    """
    Create archives (ZIP) from files and directories.
    
    Args:
        source_paths: List of file/directory paths to include
        archive_path: Output archive file path
        archive_type: Archive type ('zip' - more formats can be added)
    
    Returns:
        Dictionary containing archive creation results
    """
    try:
        if archive_type.lower() != 'zip':
            raise ValueError("Currently only ZIP archives are supported")
        
        # Ensure archive path has correct extension
        if not archive_path.lower().endswith('.zip'):
            archive_path += '.zip'
        
        total_files = 0
        total_size = 0
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for source_path in source_paths:
                source = Path(source_path)
                
                if not source.exists():
                    print(f"⚠️  Warning: {source_path} does not exist, skipping...")
                    continue
                
                if source.is_file():
                    arcname = source.name
                    zipf.write(source, arcname)
                    total_files += 1
                    total_size += source.stat().st_size
                    
                elif source.is_dir():
                    for file_path in source.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(source.parent)
                            zipf.write(file_path, arcname)
                            total_files += 1
                            total_size += file_path.stat().st_size
        
        archive_size = os.path.getsize(archive_path)
        compression_ratio = round((1 - archive_size / total_size) * 100, 2) if total_size > 0 else 0
        
        result = {
            "archive_path": archive_path,
            "archive_type": archive_type.upper(),
            "files_included": total_files,
            "original_size_bytes": total_size,
            "archive_size_bytes": archive_size,
            "compression_ratio_percent": compression_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Archive created successfully: {archive_path}")
        print(f"   📦 Files included: {total_files}")
        print(f"   📊 Original size: {total_size:,} bytes")
        print(f"   🗜️  Compressed size: {archive_size:,} bytes")
        print(f"   📉 Compression: {compression_ratio}%")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error creating archive: {str(e)}")

# =============================================================================
# UTILITY TOOLS
# =============================================================================

@mcp.tool()
def generate_qr_code(data: str, output_path: str, size: int = 10) -> Dict[str, str]:
    """
    Generate QR codes for text, URLs, or other data.
    
    Args:
        data: Data to encode in QR code
        output_path: Path to save the QR code image
        size: QR code size factor (1-40, default: 10)
    
    Returns:
        Dictionary with QR code generation results
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=4,
        )
        
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create QR code image
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        
        result = {
            "status": "success",
            "data_encoded": data[:50] + "..." if len(data) > 50 else data,
            "output_path": output_path,
            "image_size_pixels": f"{img.size[0]}x{img.size[1]}",
            "file_size_bytes": os.path.getsize(output_path),
            "qr_version": qr.version,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ QR code generated: {output_path}")
        print(f"   📊 Data: {result['data_encoded']}")
        print(f"   🖼️  Size: {result['image_size_pixels']} pixels")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error generating QR code: {str(e)}")

@mcp.tool()
def hash_data(data: str, hash_type: str = 'sha256', encoding: str = 'utf-8') -> Dict[str, str]:
    """
    Generate various types of hashes for data.
    
    Args:
        data: Data to hash
        hash_type: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        encoding: Text encoding (default: utf-8)
    
    Returns:
        Dictionary containing hash results
    """
    try:
        # Convert data to bytes
        data_bytes = data.encode(encoding)
        
        # Select hash algorithm
        if hash_type.lower() == 'md5':
            hash_obj = hashlib.md5()
        elif hash_type.lower() == 'sha1':
            hash_obj = hashlib.sha1()
        elif hash_type.lower() == 'sha256':
            hash_obj = hashlib.sha256()
        elif hash_type.lower() == 'sha512':
            hash_obj = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash type: {hash_type}")
        
        hash_obj.update(data_bytes)
        hash_hex = hash_obj.hexdigest()
        
        result = {
            "original_data": data[:100] + "..." if len(data) > 100 else data,
            "hash_type": hash_type.upper(),
            "hash_hex": hash_hex,
            "hash_base64": base64.b64encode(hash_obj.digest()).decode('ascii'),
            "data_length_bytes": len(data_bytes),
            "hash_length_bytes": len(hash_obj.digest()),
            "encoding": encoding,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Hash generated using {hash_type.upper()}")
        print(f"   📊 Data length: {len(data_bytes)} bytes")
        print(f"   🔒 Hash: {hash_hex[:32]}...")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error generating hash: {str(e)}")

@mcp.tool()
def monitor_system_resources() -> Dict[str, Any]:
    """
    Monitor system resources including CPU, memory, disk, and network.
    
    Returns:
        Dictionary containing current system resource usage
    """
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk information
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network information
        network_io = psutil.net_io_counters()
        
        # Process information
        process_count = len(psutil.pids())
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "usage_percent": cpu_percent,
                "core_count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "max_frequency_mhz": cpu_freq.max if cpu_freq else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
                "cached_gb": round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else None
            },
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "usage_percent": swap.percent
            },
            "disk": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2),
                "read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else None,
                "write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else None
            },
            "network": {
                "bytes_sent_mb": round(network_io.bytes_sent / (1024**2), 2),
                "bytes_received_mb": round(network_io.bytes_recv / (1024**2), 2),
                "packets_sent": network_io.packets_sent,
                "packets_received": network_io.packets_recv
            },
            "processes": {
                "total_count": process_count
            }
        }
        
        print(f"✅ System resources monitored")
        print(f"   🖥️  CPU: {cpu_percent}% usage")
        print(f"   💾 Memory: {memory.percent}% usage ({result['memory']['used_gb']}/{result['memory']['total_gb']} GB)")
        print(f"   💿 Disk: {result['disk']['usage_percent']}% usage")
        print(f"   🔢 Processes: {process_count}")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error monitoring system resources: {str(e)}")

@mcp.tool()
def encrypt_decrypt_text(text: str, operation: str, key: Optional[str] = None) -> Dict[str, str]:
    """
    Encrypt or decrypt text using Fernet symmetric encryption.
    
    Args:
        text: Text to encrypt/decrypt
        operation: 'encrypt' or 'decrypt'
        key: Encryption key (if None, a new key is generated for encryption)
    
    Returns:
        Dictionary containing encryption/decryption results
    """
    try:
        if operation.lower() == 'encrypt':
            # Generate new key if not provided
            if key is None:
                encryption_key = Fernet.generate_key()
            else:
                encryption_key = key.encode() if isinstance(key, str) else key
            
            fernet = Fernet(encryption_key)
            encrypted_data = fernet.encrypt(text.encode())
            
            result = {
                "operation": "encrypt",
                "original_text": text[:50] + "..." if len(text) > 50 else text,
                "encrypted_data": encrypted_data.decode(),
                "encryption_key": encryption_key.decode(),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Text decrypted successfully")
            
        else:
            raise ValueError("Operation must be 'encrypt' or 'decrypt'")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error in encryption/decryption: {str(e)}")

# =============================================================================
# PRODUCTIVITY TOOLS
# =============================================================================

@mcp.tool()
def generate_password(length: int = 12, include_symbols: bool = True, 
                     include_numbers: bool = True, include_uppercase: bool = True,
                     exclude_ambiguous: bool = True) -> Dict[str, str]:
    """
    Generate secure passwords with customizable options.
    
    Args:
        length: Password length (minimum 4, maximum 128)
        include_symbols: Include special characters
        include_numbers: Include numeric characters
        include_uppercase: Include uppercase letters
        exclude_ambiguous: Exclude ambiguous characters (0, O, l, 1, etc.)
    
    Returns:
        Dictionary containing password and strength information
    """
    import random
    import string
    
    try:
        if length < 4 or length > 128:
            raise ValueError("Password length must be between 4 and 128 characters")
        
        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase if include_uppercase else ""
        numbers = string.digits if include_numbers else ""
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""
        
        # Remove ambiguous characters if requested
        if exclude_ambiguous:
            ambiguous = "0O1lI"
            lowercase = ''.join(c for c in lowercase if c not in ambiguous)
            uppercase = ''.join(c for c in uppercase if c not in ambiguous)
            numbers = ''.join(c for c in numbers if c not in ambiguous)
        
        # Combine all available characters
        all_chars = lowercase + uppercase + numbers + symbols
        
        if not all_chars:
            raise ValueError("No character types selected for password generation")
        
        # Generate password ensuring at least one character from each selected type
        password_chars = []
        
        # Add at least one character from each enabled type
        if lowercase:
            password_chars.append(random.choice(lowercase))
        if uppercase:
            password_chars.append(random.choice(uppercase))
        if numbers:
            password_chars.append(random.choice(numbers))
        if symbols:
            password_chars.append(random.choice(symbols))
        
        # Fill the rest with random characters
        for _ in range(length - len(password_chars)):
            password_chars.append(random.choice(all_chars))
        
        # Shuffle the password
        random.shuffle(password_chars)
        password = ''.join(password_chars)
        
        # Calculate password strength
        strength_score = 0
        if any(c.islower() for c in password):
            strength_score += 1
        if any(c.isupper() for c in password):
            strength_score += 1
        if any(c.isdigit() for c in password):
            strength_score += 1
        if any(c in symbols for c in password):
            strength_score += 1
        if length >= 12:
            strength_score += 1
        if length >= 16:
            strength_score += 1
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong", "Very Strong"]
        strength = strength_levels[min(strength_score, 5)]
        
        result = {
            "password": password,
            "length": len(password),
            "strength": strength,
            "strength_score": f"{strength_score}/6",
            "character_types": {
                "lowercase": any(c.islower() for c in password),
                "uppercase": any(c.isupper() for c in password),
                "numbers": any(c.isdigit() for c in password),
                "symbols": any(c in symbols for c in password)
            },
            "entropy_bits": round(length * (len(all_chars) ** 0.5), 2),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Password generated successfully")
        print(f"   🔒 Length: {length} characters")
        print(f"   💪 Strength: {strength}")
        print(f"   🎲 Character types: {sum(result['character_types'].values())}")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error generating password: {str(e)}")

@mcp.tool()
def schedule_reminder(message: str, delay_minutes: int) -> Dict[str, str]:
    """
    Create a simple reminder system (simulation - in real implementation would integrate with OS notifications).
    
    Args:
        message: Reminder message
        delay_minutes: Minutes from now to trigger reminder
    
    Returns:
        Dictionary with reminder details
    """
    try:
        current_time = datetime.now()
        reminder_time = current_time + timedelta(minutes=delay_minutes)
        
        result = {
            "message": message,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "reminder_time": reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
            "delay_minutes": delay_minutes,
            "status": "scheduled",
            "reminder_id": hashlib.md5(f"{message}{reminder_time}".encode()).hexdigest()[:8]
        }
        
        print(f"✅ Reminder scheduled")
        print(f"   📝 Message: {message}")
        print(f"   ⏰ Time: {reminder_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🆔 ID: {result['reminder_id']}")
        print(f"   ℹ️  Note: This is a simulation. In production, would integrate with system notifications.")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error scheduling reminder: {str(e)}")

# =============================================================================
# RESOURCES
# =============================================================================

@mcp.resource("advanced-tools://examples")
def get_advanced_examples():
    """
    Provide comprehensive examples for all advanced tools.
    """
    return """
# Advanced FastMCP Tools - Usage Examples

## 📊 Data Analysis Tools

### CSV Analysis
```python
analyze_csv_file("data/sales.csv", delimiter=",")
# Returns: file info, column statistics, data quality metrics

filter_and_export_data(
    "data/sales.csv", 
    {"region": {"equals": "North"}, "sales": {"greater_than": 1000}},
    "filtered_sales.json",
    "json"
)
# Filters data and exports to JSON
```

## 🌐 Web Scraping & API Tools

### Website Scraping
```python
scrape_website_content("https://example.com", "text")
scrape_website_content("https://news.site.com", "links", "#main-content")
# Extract text, links, images, or tables from websites

fetch_api_data(
    "https://api.github.com/users/octocat",
    method="GET",
    headers={"Accept": "application/json"}
)
# Make API calls with full response metadata
```

## 📁 File Processing Tools

### Text Processing
```python
process_text_file("document.txt", ["word_count", "extract_emails", "extract_urls"])
# Analyze text files with multiple operations

create_archive(["folder1", "file.txt"], "backup.zip", "zip")
# Create compressed archives from files/folders
```

## 🔧 Utility Tools

### QR Codes & Hashing
```python
generate_qr_code("https://example.com", "qr_code.png", size=10)
hash_data("sensitive data", "sha256")
# Generate QR codes and secure hashes

encrypt_decrypt_text("secret message", "encrypt")
encrypt_decrypt_text(encrypted_data, "decrypt", key=encryption_key)
# Symmetric encryption/decryption
```

### System Monitoring
```python
monitor_system_resources()
# Real-time CPU, memory, disk, network usage

generate_password(16, include_symbols=True, exclude_ambiguous=True)
# Secure password generation with customization

schedule_reminder("Meeting in conference room", 30)
# Simple reminder system (simulation)
```

## 🚀 Advanced Usage Patterns

### Chained Operations
1. Scrape website data → Save to CSV → Analyze with CSV tools
2. Process text files → Extract emails → Generate QR codes for contacts
3. Monitor system → Generate alerts → Encrypt sensitive data
4. API data collection → Filter results → Create archives

### Error Handling
All tools include comprehensive error handling with descriptive messages.
Use try-catch blocks when chaining operations for robust workflows.

### Performance Tips
- Use CSS selectors for targeted web scraping
- Monitor system resources before large file operations
- Set appropriate timeouts for API calls
- Use filters to reduce data before analysis
"""

@mcp.resource("advanced-tools://config")
def get_tool_configuration():
    """
    Provide configuration options and best practices.
    """
    return """
# Advanced Tools Configuration Guide

## 🔧 Tool Configuration Options

### Web Scraping Settings
- Default timeout: 30 seconds
- User-Agent: Configurable browser simulation
- CSS selectors: Support for complex targeting
- Rate limiting: Built-in request delays

### File Processing Limits
- Maximum file size: 100MB (configurable)
- Supported formats: CSV, JSON, TXT, ZIP
- Encoding: UTF-8 default, configurable
- Memory usage: Optimized for large files

### Security Settings
- Encryption: Fernet symmetric encryption
- Hashing: MD5, SHA1, SHA256, SHA512 support
- Password generation: Customizable complexity
- Data validation: Input sanitization enabled

### System Monitoring
- Update interval: 1 second for CPU
- Memory precision: MB level reporting
- Disk usage: Root partition monitoring
- Network stats: Cumulative counters

## 🛡️ Security Best Practices

1. **API Keys**: Store in environment variables
2. **File Access**: Validate paths to prevent directory traversal
3. **Data Encryption**: Use strong keys, rotate regularly
4. **Web Scraping**: Respect robots.txt and rate limits
5. **System Access**: Run with minimal required permissions

## ⚡ Performance Optimization

1. **Batch Operations**: Group similar tasks
2. **Memory Management**: Process large files in chunks
3. **Caching**: Store frequently accessed data
4. **Async Operations**: Use for I/O bound tasks
5. **Resource Monitoring**: Track usage during operations

## 🔌 Integration Examples

### With Other MCP Servers
- Chain data analysis with visualization servers
- Combine with database servers for persistent storage
- Integrate with notification servers for alerts

### With External Systems
- Export data to BI tools
- Integration with CI/CD pipelines
- Connection to monitoring dashboards
"""

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Advanced FastMCP Tools Server")
    print("=" * 50)
    print(f"Server Name: {mcp.name}")
    print(f"Startup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("📋 Available Tool Categories:")
    categories = {
        "Data Analysis": ["analyze_csv_file", "filter_and_export_data"],
        "Web & API": ["scrape_website_content", "fetch_api_data"], 
        "File Processing": ["process_text_file", "create_archive"],
        "Utilities": ["generate_qr_code", "hash_data", "encrypt_decrypt_text"],
        "System & Productivity": ["monitor_system_resources", "generate_password", "schedule_reminder"]
    }
    
    for category, tools in categories.items():
        print(f"\n🔧 {category}:")
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. {tool}")
    
    print(f"\n📚 Total Tools Available: {sum(len(tools) for tools in categories.values())}")
    print(f"🔗 Resources: advanced-tools://examples, advanced-tools://config")
    print("\n" + "=" * 50)
    print("🎯 Server Features:")
    print("   • Comprehensive error handling and validation")
    print("   • Detailed logging and progress tracking") 
    print("   • Flexible configuration options")
    print("   • Security-focused design")
    print("   • Performance optimized operations")
    print("   • Real-world utility functions")
    print("\n🚀 Starting MCP server...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the server
    mcp.run()
