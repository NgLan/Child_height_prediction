import pandas as pd
import numpy as np
import os

from config import WHO_REF_DIR
# --- Configuration for WHO reference files ---
# Length/Height-for-Age (L/HFA) - Age in Days
LHFA_BOYS_FILE = os.path.join(WHO_REF_DIR, "WHO-lhfa-boys-zscore-expanded-tables-0-5y.csv")
LHFA_GIRLS_FILE = os.path.join(WHO_REF_DIR, "WHO-lhfa-girls-zscore-expanded-tables-0-5y.csv")

# Weight-for-Age (WFA) - Age in Days
WFA_BOYS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfa-boys-zscore-expanded-tables-0-5y.csv")
WFA_GIRLS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfa-girls-zscore-expanded-tables-0-5y.csv")

# Weight-for-Length (WFL) - Length in cm (typically for < 2 years)
WFL_BOYS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfl-boys-zscore-expanded-tables-0-5y.csv") 
WFL_GIRLS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfl-girls-zscore-expanded-tables-0-5y.csv")

# Weight-for-Height (WFH) - Height in cm (typically for >= 2 years)
WFH_BOYS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfh-boys-zscore-expanded-tables-0-5y.csv") 
WFH_GIRLS_FILE = os.path.join(WHO_REF_DIR, "WHO-wfh-girls-zscore-expanded-tables-0-5y.csv")


# Column names in your WHO CSV files
AGE_COL_WHO = "Day"
LENGTH_COL_WHO = "Length" # For WFL tables
HEIGHT_COL_WHO = "Height" # For WFH tables

L_COL_WHO = "L"
M_COL_WHO = "M"
S_COL_WHO = "S"

# --- Load WHO Reference Data ---
def load_who_references():
    """Loads all necessary WHO reference tables."""
    refs = {}
    files_to_load = {
        'lhfa_boys': LHFA_BOYS_FILE,
        'lhfa_girls': LHFA_GIRLS_FILE,
        'wfa_boys': WFA_BOYS_FILE,
        'wfa_girls': WFA_GIRLS_FILE,
        'wfl_boys': WFL_BOYS_FILE, # Weight-for-Length
        'wfl_girls': WFL_GIRLS_FILE,# Weight-for-Length
        'wfh_boys': WFH_BOYS_FILE,  # Weight-for-Height
        'wfh_girls': WFH_GIRLS_FILE, # Weight-for-Height
    }
    all_loaded = True
    for key, file_path in files_to_load.items():
        try:
            # Xử lý BOM (Byte Order Mark) nếu có trong file CSV (ký tự ﻿ ở đầu)
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                refs[key] = pd.read_csv(f)
            # Kiểm tra xem cột đầu tiên có BOM không và loại bỏ nếu cần
            first_col_name = refs[key].columns[0]
            if first_col_name.startswith('\ufeff'):
                refs[key].rename(columns={first_col_name: first_col_name.lstrip('\ufeff')}, inplace=True)

        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy tệp tham chiếu WHO: {file_path}")
            refs[key] = pd.DataFrame() # Trả về DataFrame rỗng nếu không tìm thấy tệp
            all_loaded = False
        except Exception as e:
            print(f"Lỗi khi tải tệp {file_path}: {e}")
            refs[key] = pd.DataFrame()
            all_loaded = False

    if all_loaded:
        print("Tất cả dữ liệu tham chiếu WHO đã được tải thành công.")
    else:
        print("CẢNH BÁO: Một số tệp tham chiếu WHO không được tải. Tính toán Z-score có thể không chính xác.")
    return refs

# --- LMS Value Interpolation ---
def get_lms_values(indicator_value, ref_df, indicator_col_name):
    """
    Gets L, M, S values for a given indicator value (age or height/length) from a reference DataFrame.
    Performs linear interpolation if exact match not found.
    Handles cases where indicator_value is outside the range of the reference table.
    """
    if ref_df.empty or indicator_col_name not in ref_df.columns:
        # print(f"Cảnh báo: DataFrame tham chiếu rỗng hoặc thiếu cột {indicator_col_name}")
        return None, None, None
    if L_COL_WHO not in ref_df.columns or M_COL_WHO not in ref_df.columns or S_COL_WHO not in ref_df.columns:
        # print(f"Cảnh báo: DataFrame tham chiếu thiếu các cột LMS.")
        return None, None, None

    ref_df = ref_df.sort_values(by=indicator_col_name)
    ref_df[indicator_col_name] = pd.to_numeric(ref_df[indicator_col_name], errors='coerce')
    ref_df = ref_df.dropna(subset=[indicator_col_name]) # Loại bỏ NaN trong cột chỉ số

    if ref_df.empty: # Kiểm tra lại sau khi dropna
        return None, None, None

    # Exact match
    match = ref_df[ref_df[indicator_col_name] == indicator_value]
    if not match.empty:
        return match.iloc[0][L_COL_WHO], match.iloc[0][M_COL_WHO], match.iloc[0][S_COL_WHO]

    min_indicator = ref_df[indicator_col_name].min()
    max_indicator = ref_df[indicator_col_name].max()

    if indicator_value < min_indicator:
        closest_row = ref_df.iloc[0]
        return closest_row[L_COL_WHO], closest_row[M_COL_WHO], closest_row[S_COL_WHO]
    elif indicator_value > max_indicator:
        closest_row = ref_df.iloc[-1]
        return closest_row[L_COL_WHO], closest_row[M_COL_WHO], closest_row[S_COL_WHO]
    else:
        # Đảm bảo các cột LMS là số trước khi nội suy
        for col in [L_COL_WHO, M_COL_WHO, S_COL_WHO]:
            ref_df[col] = pd.to_numeric(ref_df[col], errors='coerce')
        ref_df = ref_df.dropna(subset=[L_COL_WHO, M_COL_WHO, S_COL_WHO])
        if ref_df.empty: return None, None, None

        l_val = np.interp(indicator_value, ref_df[indicator_col_name], ref_df[L_COL_WHO])
        m_val = np.interp(indicator_value, ref_df[indicator_col_name], ref_df[M_COL_WHO])
        s_val = np.interp(indicator_value, ref_df[indicator_col_name], ref_df[S_COL_WHO])
        return l_val, m_val, s_val

# --- Z-score Calculation (as per WHO formula) ---
def _calculate_sd_val(M, L, S, z_cutoff):
    if pd.isna(M) or pd.isna(L) or pd.isna(S): return np.nan
    if L == 0:
        return M * np.exp(S * z_cutoff)
    base = 1 + L * S * z_cutoff
    if base <= 0:
        # Theo WHO Anthro manual, nếu base <= 0, z-score được tính bằng công thức đơn giản hơn
        # Hoặc giá trị được coi là cực đoan và không tính SD value theo cách này.
        # For simplicity, returning NaN, but specific guidelines might exist.
        # print(f"Cảnh báo: Base <=0 khi tính SD value. M={M}, L={L}, S={S}, z_cutoff={z_cutoff}, base={base}")
        return np.nan
    # Thêm epsilon nhỏ để tránh lỗi chia cho 0 nếu L rất gần 0
    return M * (base ** (1 / (L if L != 0 else 1e-9) ))


def calculate_zscore_who(y, L, M, S):
    if pd.isna(y) or pd.isna(L) or pd.isna(M) or pd.isna(S) or M <= 0 or S <= 0:
        return np.nan

    # Step 1: Calculate initial z_ind
    if L == 0:
        if y <= 0: return np.nan # y phải dương
        z_ind = np.log(y / M) / S
    else:
        if y <= 0: return np.nan # y phải dương
        # Thêm epsilon nhỏ để tránh lỗi chia cho 0 nếu L rất gần 0
        z_ind = (((y / M)**L) - 1) / (S * (L if L != 0 else 1e-9))
    
    if pd.isna(z_ind):
        return np.nan

    # Step 2: Compute final z-score (z_ind_star)
    # Theo WHO, việc điều chỉnh z-score chỉ áp dụng cho HFA và WFH/WFL
    # WFA không điều chỉnh z-score > 3 hoặc < -3.
    # Tuy nhiên, công thức bạn cung cấp là công thức chung, nên tôi sẽ giữ nguyên.
    # Nếu chỉ áp dụng cho HFA và WFH/WFL, cần thêm tham số 'indicator_type'
    
    z_ind_star = z_ind # Mặc định

    if abs(z_ind) > 3: # Chỉ điều chỉnh nếu z_ind vượt quá +/-3
        # Các giá trị SD2, SD3, SD-2, SD-3 được tính từ bảng tham chiếu, không phải từ _calculate_sd_val
        # Tuy nhiên, các file bạn cung cấp có các cột SDxneg/SDx.
        # Nếu không có, chúng ta mới dùng _calculate_sd_val.
        # Hiện tại, công thức bạn đưa ra là TÍNH TOÁN chúng.
        
        SD3pos = _calculate_sd_val(M, L, S, 3)
        SD3neg = _calculate_sd_val(M, L, S, -3)
        SD2pos = _calculate_sd_val(M, L, S, 2) # SD2pos
        SD2neg = _calculate_sd_val(M, L, S, -2) # SD2neg

        if pd.isna(SD3pos) or pd.isna(SD3neg) or pd.isna(SD2pos) or pd.isna(SD2neg):
            # print(f"Không thể tính SD cutoffs cho y={y}, L={L}, M={M}, S={S}")
            return z_ind # Trả về z_ind ban đầu nếu không tính được SD cutoffs

        if z_ind > 3:
            SD23pos = SD3pos - SD2pos # Đây là M[1+LS(3)]^(1/L) - M[1+LS(2)]^(1/L)
            if SD23pos == 0 or pd.isna(SD23pos):
                z_ind_star = 3 
            else:
                z_ind_star = 3 + ((y - SD3pos) / SD23pos)
        elif z_ind < -3:
            SD23neg = SD2neg - SD3neg # Đây là M[1+LS(-2)]^(1/L) - M[1+LS(-3)]^(1/L)
            if SD23neg == 0 or pd.isna(SD23neg): 
                 z_ind_star = -3
            else:
                z_ind_star = -3 + ((y - SD3neg) / SD23neg)
        # else trường hợp abs(z_ind) <=3 đã được xử lý bằng z_ind_star = z_ind
            
    return z_ind_star

# --- Specific Z-score Getters ---
def get_hfa_zscore(height_cm, age_days, sex, who_refs):
    if sex == 1: # Male
        ref_df = who_refs.get('lhfa_boys')
    elif sex == 0: # Female
        ref_df = who_refs.get('lhfa_girls')
    else:
        return np.nan
    
    if ref_df is None or ref_df.empty: return np.nan
    L, M, S = get_lms_values(age_days, ref_df, AGE_COL_WHO)
    if L is None or M is None or S is None: return np.nan
    return calculate_zscore_who(height_cm, L, M, S)

def get_wfa_zscore(weight_kg, age_days, sex, who_refs):
    if sex == 1: # Male
        ref_df = who_refs.get('wfa_boys')
    elif sex == 0: # Female
        ref_df = who_refs.get('wfa_girls')
    else:
        return np.nan

    if ref_df is None or ref_df.empty: return np.nan
    L, M, S = get_lms_values(age_days, ref_df, AGE_COL_WHO)
    if L is None or M is None or S is None: return np.nan
    # WFA z-scores are typically not adjusted for extreme values in the same way as HFA/WFH
    # So we might directly use the initial z_ind if that's the guideline.
    # For now, using the general formula provided.
    return calculate_zscore_who(weight_kg, L, M, S)

def get_wfh_zscore(weight_kg, measured_val, sex, age_days, who_refs):
    """
    Calculates Weight-for-Height/Length Z-score.
    measured_val: height_cm or length_cm.
    age_days: used to determine if length or height reference should be used.
    """
    # WHO: Length for < 24 months (approx 730 days), Height for >= 24 months
    # Or, Length if recumbent length < 87 cm, Height if standing height >= 87 cm
    # Chúng ta dùng age_days để quyết định.
    # Và giả sử các file WFL/WFH của bạn đã có sẵn.
    
    # Ngưỡng tuổi để chuyển từ Length sang Height (ví dụ: 2 tuổi = 730 ngày)
    # Ngưỡng chiều dài/cao để chuyển (ví dụ: 87 cm)
    # WHO thường dùng chiều dài (Length) cho trẻ dưới 2 tuổi.
    # Nếu tuổi >= 2 tuổi, dùng chiều cao (Height).
    # Một số hướng dẫn khác có thể dựa trên chiều dài/cao đo được (ví dụ <87cm dùng length).
    # Ở đây, chúng ta sẽ dùng tuổi để quyết định bảng nào.
    
    use_length_ref = age_days < 730 # Dưới 2 tuổi (730 ngày) dùng Length

    if sex == 1: # Male
        ref_df = who_refs.get('wfl_boys') if use_length_ref else who_refs.get('wfh_boys')
        indicator_col = LENGTH_COL_WHO if use_length_ref else HEIGHT_COL_WHO
    elif sex == 0: # Female
        ref_df = who_refs.get('wfl_girls') if use_length_ref else who_refs.get('wfh_girls')
        indicator_col = LENGTH_COL_WHO if use_length_ref else HEIGHT_COL_WHO
    else:
        return np.nan
    
    if ref_df is None or ref_df.empty:
        print(f"CẢNH BÁO: Thiếu bảng tham chiếu WFH/WFL cho sex={sex}, use_length={use_length_ref}")
        return np.nan

    L, M, S = get_lms_values(measured_val, ref_df, indicator_col)
    if L is None or M is None or S is None: return np.nan
    return calculate_zscore_who(weight_kg, L, M, S)


# --- Classification Logic (giữ nguyên) ---
def classify_stunting(hfa_zscore):
    if pd.isna(hfa_zscore): return "HFA: Unknown"
    if hfa_zscore < -3: return "HFA: Severely stunted"
    if hfa_zscore < -2: return "HFA: Stunted"
    return "HFA: Normal height-for-age"

def classify_underweight(wfa_zscore):
    if pd.isna(wfa_zscore): return "WFA: Unknown"
    if wfa_zscore < -3: return "WFA: Severely underweight"
    if wfa_zscore < -2: return "WFA: Underweight"
    return "WFA: Normal weight-for-age"

def classify_wasting_overweight(wfh_zscore):
    if pd.isna(wfh_zscore): return "WFH/L: Unknown"
    
    status = []
    # Wasting
    if wfh_zscore < -3: status.append("WFH/L: Severely wasted (Refer for urgent specialized care)")
    elif wfh_zscore < -2: status.append("WFH/L: Wasted")
    
    # Overweight/Obesity
    if wfh_zscore > 3: status.append("WFH/L: Obese")
    elif wfh_zscore > 2: status.append("WFH/L: Overweight")
    elif wfh_zscore > 1: status.append("WFH/L: Possible risk of overweight")

    if not status:
        return "WFH/L: Normal weight-for-height/length"
    return "; ".join(status)

# --- Main analysis function to be called from main.py ---
def analyze_child_growth_from_predictions(subjid, sex, predictions_h_ts, predictions_w_ts, who_refs, start_date_ref):
    if who_refs is None:
        print(f"Subjid {subjid}: WHO references not loaded. Skipping Z-score analysis.")
        return

    print(f"\n--- Z-score Analysis for Subjid: {subjid} (Sex: {'Female' if sex == 0 else 'Male'}) ---")

    if predictions_h_ts is None or len(predictions_h_ts) == 0 or \
       predictions_w_ts is None or len(predictions_w_ts) == 0:
        print("Not enough prediction data to analyze.")
        return

    # Lấy điểm dự đoán cuối cùng
    # Cân nhắc: Có thể bạn muốn phân tích TẤT CẢ các điểm dự đoán, không chỉ điểm cuối cùng.
    # Hoặc một điểm cụ thể trong tương lai. Hiện tại, lấy điểm cuối cùng.
    last_time_h = predictions_h_ts.end_time()
    predicted_height = predictions_h_ts.values()[-1][0] # Lấy giá trị cuối cùng
    
    # Tìm giá trị cân nặng tại cùng thời điểm hoặc gần nhất
    try:
        # Cố gắng lấy giá trị tại chính xác thời điểm đó
        predicted_weight_series_at_time = predictions_w_ts.slice_intersect(predictions_h_ts.slice(last_time_h, last_time_h))
        if len(predicted_weight_series_at_time) > 0:
            predicted_weight = predicted_weight_series_at_time.values()[-1][0]
        else:
            # Nếu không có, lấy giá trị cuối cùng của chuỗi cân nặng (có thể không hoàn toàn khớp thời gian)
            print(f"Subjid {subjid}: No exact matching weight prediction for the last height prediction time. Using last available weight.")
            predicted_weight = predictions_w_ts.values()[-1][0]
    except Exception as e:
        print(f"Subjid {subjid}: Error aligning weight prediction: {e}. Using last available weight.")
        predicted_weight = predictions_w_ts.values()[-1][0] # Fallback

    agedays = (last_time_h - start_date_ref).days
    
    print(f"Analyzing at Agedays: {agedays:.0f}, Predicted Height/Length: {predicted_height:.1f} cm, Predicted Weight: {predicted_weight:.2f} kg")

    hfa_z = get_hfa_zscore(predicted_height, agedays, sex, who_refs)
    wfa_z = get_wfa_zscore(predicted_weight, agedays, sex, who_refs)
    
    # Đối với WFH/L, chúng ta truyền `agedays` để quyết định dùng bảng Length hay Height
    wfh_z = get_wfh_zscore(predicted_weight, predicted_height, sex, agedays, who_refs)

    print(f"  HFA Z-score: {hfa_z:.2f}" if not pd.isna(hfa_z) else "  HFA Z-score: N/A")
    print(f"  WFA Z-score: {wfa_z:.2f}" if not pd.isna(wfa_z) else "  WFA Z-score: N/A")
    print(f"  WFH/L Z-score: {wfh_z:.2f}" if not pd.isna(wfh_z) else "  WFH/L Z-score: N/A")

    print(f"  Classification - Stunting: {classify_stunting(hfa_z)}")
    print(f"  Classification - Underweight: {classify_underweight(wfa_z)}")
    print(f"  Classification - Wasting/Overweight: {classify_wasting_overweight(wfh_z)}")

if __name__ == '__main__':
    print("Testing Z-score calculator module...")
    
    # Tải dữ liệu tham chiếu WHO thực tế (đảm bảo đường dẫn đúng)
    # Đặt các file CSV của WHO vào thư mục ../data/who_references/ so với vị trí file này
    # Hoặc điều chỉnh BASE_PATH_WHO
    actual_who_refs = load_who_references()

    if actual_who_refs and not actual_who_refs['lhfa_girls'].empty : # Kiểm tra xem có tải được không
        start_test_date = pd.Timestamp('2000-01-01')
        
        # Test Case 1: Bé gái, 90 ngày tuổi, cao 60cm, nặng 5.5kg
        # Tạo TimeSeries giả lập cho dự đoán
        pred_h_vals = np.array([58.0, 59.0, 60.0]) # Giả sử 3 điểm dự đoán
        pred_w_vals = np.array([5.0, 5.2, 5.5])
        
        # Tạo time index tương ứng với 30, 60, 90 ngày
        time_idx_test = pd.date_range(start=start_test_date + pd.Timedelta(days=30), periods=3, freq='30D')

        dummy_height_ts = TimeSeries.from_times_and_values(time_idx_test, pred_h_vals)
        dummy_weight_ts = TimeSeries.from_times_and_values(time_idx_test, pred_w_vals)

        analyze_child_growth_from_predictions(
            subjid="test_girl_01", 
            sex=0, # Female
            predictions_h_ts=dummy_height_ts,
            predictions_w_ts=dummy_weight_ts,
            who_refs=actual_who_refs,
            start_date_ref=start_test_date 
        )

        # Test Case 2: Bé trai, 365 ngày tuổi (1 tuổi), cao 75cm, nặng 9.5kg
        pred_h_vals_b = np.array([73.0, 74.0, 75.0])
        pred_w_vals_b = np.array([9.0, 9.2, 9.5])
        time_idx_test_b = pd.date_range(start=start_test_date + pd.Timedelta(days=305), periods=3, freq='30D') # Kết thúc ở 365 ngày

        dummy_height_ts_b = TimeSeries.from_times_and_values(time_idx_test_b, pred_h_vals_b)
        dummy_weight_ts_b = TimeSeries.from_times_and_values(time_idx_test_b, pred_w_vals_b)
        
        analyze_child_growth_from_predictions(
            subjid="test_boy_01", 
            sex=1, # Male
            predictions_h_ts=dummy_height_ts_b,
            predictions_w_ts=dummy_weight_ts_b,
            who_refs=actual_who_refs,
            start_date_ref=start_test_date
        )
        
        # Test case for L=0 (using a modified M, S for visibility)
        print("\nTesting L=0 case (manual values):")
        y_test_l0 = 60
        L_test_l0 = 0
        M_test_l0 = 65
        S_test_l0 = 0.09
        z_l0 = calculate_zscore_who(y_test_l0, L_test_l0, M_test_l0, S_test_l0)
        print(f"y={y_test_l0}, L={L_test_l0}, M={M_test_l0}, S={S_test_l0} => Z-score={z_l0:.2f} (Expected approx -0.86)")

        print("\nTesting extreme z_ind > 3 case (manual values):")
        y_extreme_pos = 90 
        L_extreme, M_extreme, S_extreme = -0.2, 65, 0.09 
        z_extreme_pos = calculate_zscore_who(y_extreme_pos, L_extreme, M_extreme, S_extreme)
        print(f"y={y_extreme_pos}, L={L_extreme}, M={M_extreme}, S={S_extreme} => Z-score={z_extreme_pos:.2f}")

        print("\nTesting extreme z_ind < -3 case (manual values):")
        y_extreme_neg = 40
        z_extreme_neg = calculate_zscore_who(y_extreme_neg, L_extreme, M_extreme, S_extreme)
        print(f"y={y_extreme_neg}, L={L_extreme}, M={M_extreme}, S={S_extreme} => Z-score={z_extreme_neg:.2f}")
    else:
        print("Không thể chạy test do thiếu dữ liệu tham chiếu WHO thực tế.")