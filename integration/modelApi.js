/** V4 adapter. Wire this to your frontend; it does not invent Fair Price data. */
export async function getAirPulseInsight({ apiBase, history, asOf, signal }) {
  if (!Array.isArray(history) || !asOf) {
    throw new Error("Replay needs comparable quote history and an explicit asOf");
  }
  const response = await fetch(`${apiBase.replace(/\/$/, "")}/insight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ mode: "replay", as_of: asOf, history }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail?.message || "Không lấy được dự báo");
  if (result.schema_version !== "airpulse.insight.v4") throw new Error("Sai phiên bản API");
  return result;
}

export function toUiState(result) {
  const reasons = {
    INSUFFICIENT_HISTORY: "Chưa đủ lịch sử giá",
    STALE_QUOTES: "Chưa có báo giá đủ mới",
    OUT_OF_SCOPE: "Tuyến/hãng chưa có đủ dữ liệu hỗ trợ",
    DEPARTURE_PASSED: "Chuyến bay đã khởi hành",
    OUTSIDE_SUPPORTED_PURCHASE_WINDOW: "Ngoài khoảng thời gian model hỗ trợ",
    NOT_VALIDATED_FOR_LIVE: "Model nghiên cứu chưa được kiểm chứng để tư vấn hiện tại",
  };
  if (result.status !== "OK") {
    return { state: "unavailable", message: reasons[result.status] || result.status,
      action: null, forecasts: [], replay: result.mode === "replay" };
  }
  const r = result.recommendation;
  return { state: "ready", replay: result.mode === "replay", asOf: r.as_of,
    latestQuoteAt: r.latest_quote_at, action: r.action,
    actionLabel: r.action === "WAIT_AND_RECHECK" ? "Chờ và kiểm tra lại" : "Mua ngay theo chính sách hiện tại",
    explanation: r.reason_code === "NO_VALIDATED_WAIT_ADVANTAGE"
      ? "Chưa xác nhận được lợi thế của việc chờ; điều này không có nghĩa giá chắc chắn không giảm."
      : r.reason_code,
    nextCheckAt: r.next_check_at, recommendedBuyAt: r.recommended_buy_at,
    forecasts: result.forecast_slots, warnings: result.warnings,
    priceIsLive: false, priceIsPayableQuote: false };
}
