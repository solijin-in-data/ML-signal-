import requests
import pandas as pd

# 1. Đường link API chính xác bạn vừa tìm được
url = "https://api-feature.sstock.vn/api/v1/prices/history?symbol=CTD&from=2008-12-31&to=2026-06-22"

# 2. Cấu hình Headers (Bắt buộc phải có để giả lập trình duyệt, tránh bị hệ thống sstock chặn lỗi 403)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://sstock.vn/",
    "Origin": "https://sstock.vn",
    "Cookie": "__Secure-better-auth.session_token=fbps3YOiP2LRRYdPWILrEBDY6x1uqYaA.tRbJKYIZDPKMkysXoDSRfSVQtgP8OEZj4uzV9Tdy5yw%3D; sstock.current_company_full_info={%22code%22:%22CTD%22%2C%22label%22:%22C%C3%B4ng%20ty%20C%E1%BB%95%20ph%E1%BA%A7n%20X%C3%A2y%20d%E1%BB%B1ng%20COTECCONS%22%2C%22value%22:%22CTD%22%2C%22sector%22:%22X%C3%A2y%20d%E1%BB%B1ng%22%2C%22sectorId%22:%2241%22}; __Secure-better-auth.session_data=eyJzZXNzaW9uIjp7InNlc3Npb24iOnsiZXhwaXJlc0F0IjoiMjAyNi0wNi0yOVQwMjo0MzozMC4zMDNaIiwidG9rZW4iOiJmYnBzM1lPaVAyTFJSWWRQV0lMckVCRFk2eDF1cVlhQSIsImNyZWF0ZWRBdCI6IjIwMjYtMDYtMjJUMDI6NDM6MzAuMzAzWiIsInVwZGF0ZWRBdCI6IjIwMjYtMDYtMjJUMDI6NDM6MzAuMzAzWiIsImlwQWRkcmVzcyI6IjEwLjQyLjAuMTQ3IiwidXNlckFnZW50IjoiTW96aWxsYS81LjAgKFdpbmRvd3MgTlQgMTAuMDsgV2luNjQ7IHg2NCkgQXBwbGVXZWJLaXQvNTM3LjM2IChLSFRNTCwgbGlrZSBHZWNrbykgQ2hyb21lLzE0OS4wLjAuMCBTYWZhcmkvNTM3LjM2IEVkZy8xNDkuMC4wLjAiLCJ1c2VySWQiOiI1N2J4ZTBRQWVJMWhVbDFrSnJlRVB2d011ellxdHgxbyIsImltcGVyc29uYXRlZEJ5IjpudWxsLCJhY3RpdmVPcmdhbml6YXRpb25JZCI6bnVsbCwiYWN0aXZlVGVhbUlkIjpudWxsLCJpZCI6ImZ5SkxlQ3VnM28wZzI1dlI1c2htMGZmTUF5ZzZ3eDRCIn0sInVzZXIiOnsibmFtZSI6IlFVw4JOIE5HVVnhu4ROIEFOSCIsImVtYWlsIjoicXVhbm5ndXllbi4zMTI0MTAyNTQzNEBzdC51ZWguZWR1LnZuIiwiZW1haWxWZXJpZmllZCI6dHJ1ZSwiaW1hZ2UiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKNHVqRS1TbGd3Zks4YjNBWkRXdkVHNGc4T2dzX2dYcXNLMGxaaGpTX01rVVBnVGc9czk2LWMiLCJjcmVhdGVkQXQiOiIyMDI2LTAzLTE5VDIxOjIxOjE3LjM4OVoiLCJ1cGRhdGVkQXQiOiIyMDI2LTAzLTE5VDIxOjIxOjE3LjM4OVoiLCJ1c2VybmFtZSI6bnVsbCwiZGlzcGxheVVzZXJuYW1lIjpudWxsLCJyb2xlIjoidXNlciIsImJhbm5lZCI6ZmFsc2UsImJhblJlYXNvbiI6bnVsbCwiYmFuRXhwaXJlcyI6bnVsbCwidXNlclR5cGUiOm51bGwsImRpc3BsYXlQaG9uZU51bWJlciI6bnVsbCwiaWQiOiI1N2J4ZTBRQWVJMWhVbDFrSnJlRVB2d011ellxdHgxbyJ9LCJ1cGRhdGVkQXQiOjE3ODIxMTA4NTI3ODQsInZlcnNpb24iOiIxIn0sImV4cGlyZXNBdCI6MTc4MjExMTE1Mjc4NCwic2lnbmF0dXJlIjoidUtkZURCcTBjTWEzNlFWSUN2LU92ZFE0S3B6RkJ4ZFVKdUx5eU13SEZuNCJ9; ph_phc_2O2eCgo6AOpwUykoQ5ufJGvaahcsg9cOPCMp4sZwSMh_posthog=%7B%22%24device_id%22%3A%22019e1743-e90e-781f-b627-8283fd9fc9cc%22%2C%22distinct_id%22%3A%22019e1743-e90e-781f-b627-8283fd9fc9cc%22%2C%22%24sesid%22%3A%5B1782111148210%2C%22019eedff-f7f8-7cb3-af05-c266a2ac5745%22%2C1782109435895%5D%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22https%3A%2F%2Fwww.bing.com%2F%22%2C%22u%22%3A%22https%3A%2F%2Fsstock.vn%2F%22%7D%2C%22%24user_state%22%3A%22anonymous%22%7D"
}

try:
    print("🚀 Đang kết nối tới API của sstock để tải dữ liệu...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        
        # Kiểm tra nếu dữ liệu nằm trong key 'data' hoặc trả về dạng danh sách thẳng
        data_list = json_data.get('data', json_data) if isinstance(json_data, dict) else json_data
        
        # 3. Chuyển đổi dữ liệu JSON thành Bảng (DataFrame)
        df = pd.DataFrame(data_list)
        
        # 4. Lọc các cột quan trọng nhất để bảng Excel gọn gàng
        columns_to_keep = ['date', 'symbol', 'marketCap', 'close', 'volume', 'open', 'high', 'low']
        # Chỉ giữ lại các cột nếu chúng thực sự tồn tại trong dữ liệu trả về
        df_filtered = df[[col for col in columns_to_keep if col in df.columns]].copy()
        
        # 5. Xử lý định dạng dữ liệu cho đẹp
        if 'marketCap' in df_filtered.columns:
            # Thêm cột Vốn hóa tính theo đơn vị Tỷ đồng cho dễ đọc
            df_filtered['marketCap_TyDong'] = df_filtered['marketCap'] / 1_000_000_000
            
        if 'date' in df_filtered.columns:
            # Sắp xếp ngày tháng từ cũ nhất đến mới nhất
            df_filtered = df_filtered.sort_values(by='date', ascending=True)
        
        # 6. Xuất dữ liệu thẳng ra file Excel
        output_file = "CTD_lich_su_von_hoa.xlsx"
        df_filtered.to_excel(output_file, index=False)
        
        print("\n🎉 THÀNH CÔNG RỰC RỠ!")
        print(f"📊 Đã lưu toàn bộ dữ liệu vào file: '{output_file}'")
        print("👀 Dưới đây là bản xem trước 5 dòng dữ liệu gần nhất:")
        print(df_filtered.tail())
        
    else:
        print(f"❌ Kết nối thất bại! Mã lỗi HTTP: {response.status_code}")
        print("Mẹo: Nếu gặp lỗi 401 hoặc 403, bạn cần mở tab Network, copy chuỗi 'Cookie' trong Request Headers rồi dán vào biến headers của code.")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi trong quá trình xử lý: {e}")