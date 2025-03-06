/**
 * Map click identification workflow.
 *
 * This file demonstrates the engineering flow behind "click a roof on the map
 * and convert it into a structured photovoltaic lead". API keys, persistence,
 * business records and full production implementations are intentionally omitted.
 */

const IDENTIFY_DEFAULTS = {
  nearbyRadiusMeters: 160,
  maxPoiCandidates: 8,
  minVisionConfidence: 0.55,
  minContactConfidence: 0.6,
  minLeadScoreForFollowUp: 70,
};

/**
 * @param {Object} params
 * @param {number} params.lat Display-layer latitude.
 * @param {number} params.lng Display-layer longitude.
 * @param {number} params.zoom Current map zoom.
 * @param {"gcj02"|"wgs84"} params.coordinateSystem Coordinate system used by the map layer.
 * @param {Object} params.adapters Service adapters owned by the backend gateway.
 * @returns {Promise<Object>} Structured lead identification result.
 */
async function identifyMapClick(params) {
  const {
    lat,
    lng,
    zoom,
    coordinateSystem = "gcj02",
    adapters,
  } = params;

  assertValidCoordinate({ lat, lng });
  assertAdapters(adapters);

  const displayPoint = { lat, lng };
  const queryPoint = await normalizeToGcj02({
    point: displayPoint,
    source: coordinateSystem,
    coordinateService: adapters.coordinateService,
  });

  const [geoResult, poiResult, visionResult] = await Promise.allSettled([
    adapters.mapService.reverseGeocode(queryPoint),
    adapters.mapService.searchNearby({
      ...queryPoint,
      radius: IDENTIFY_DEFAULTS.nearbyRadiusMeters,
      limit: IDENTIFY_DEFAULTS.maxPoiCandidates,
    }),
    adapters.visionService.detectRoofResource({
      imageTile: buildSatelliteTileRequest({ point: displayPoint, zoom }),
      coordinate: queryPoint,
      zoom,
    }),
  ]);

  const place = normalizePlace(readSettledValue(geoResult));
  const poiCandidates = normalizePoiCandidates(readSettledValue(poiResult));
  const roofVision = normalizeRoofVision(readSettledValue(visionResult));
  const matchedPoi = selectBestPoi({
    place,
    poiCandidates,
    coordinate: queryPoint,
    roofVision,
  });

  const roofProfile = buildRoofProfile({
    place,
    matchedPoi,
    roofVision,
  });

  const leadScore = calculateLeadScore({
    roofArea: roofProfile.estimatedRoofArea,
    buildingType: roofProfile.buildingType,
    hasVerifiedContact: roofProfile.contactAvailable,
    pvStatus: roofProfile.pvStatus,
    visionConfidence: roofVision.confidence,
  });

  return {
    coordinate: {
      display: displayPoint,
      query: queryPoint,
      querySystem: "gcj02",
    },
    lead: {
      displayName: roofProfile.displayName,
      addressLevel: roofProfile.addressLevel,
      contactMasked: maskContact(matchedPoi.contact),
      contactAvailable: roofProfile.contactAvailable,
      buildingType: roofProfile.buildingType,
      estimatedRoofArea: roofProfile.estimatedRoofArea,
      pvStatus: roofProfile.pvStatus,
      score: leadScore,
      priority: leadScore >= IDENTIFY_DEFAULTS.minLeadScoreForFollowUp ? "A" : "B",
    },
    evidence: {
      source: buildEvidenceSource({ geoResult, poiResult, visionResult }),
      poiMatchConfidence: matchedPoi.matchConfidence,
      visionConfidence: roofVision.confidence,
      reviewRequired: shouldRequireManualReview({ matchedPoi, roofVision }),
    },
  };
}

function buildRoofProfile({ place, matchedPoi, roofVision }) {
  const buildingType = normalizeBuildingType(matchedPoi.type || place.type);
  const estimatedRoofArea = roofVision.roofArea || estimateRoofAreaByType(buildingType);
  const pvStatus = resolvePvStatus(roofVision);

  return {
    displayName: matchedPoi.name || place.name || "Map Selected Location",
    addressLevel: summarizeAddressLevel(matchedPoi.address || place.address),
    contactAvailable: normalizeContact(matchedPoi.contact).length > 0,
    buildingType,
    estimatedRoofArea,
    pvStatus,
  };
}

function selectBestPoi({ place, poiCandidates, coordinate, roofVision }) {
  const scored = poiCandidates.map((poi) => ({
    ...poi,
    matchConfidence: calculatePoiMatchConfidence({
      poi,
      place,
      coordinate,
      roofVision,
    }),
  }));

  scored.sort((a, b) => b.matchConfidence - a.matchConfidence);

  return scored[0] || {
    name: place.name,
    address: place.address,
    type: place.type,
    contact: "",
    matchConfidence: 0,
  };
}

function calculatePoiMatchConfidence({ poi, place, coordinate, roofVision }) {
  const distanceScore = scoreDistance(poi.location, coordinate);
  const addressScore = scoreTextOverlap(poi.address, place.address);
  const contactScore = normalizeContact(poi.contact).length > 0 ? 0.15 : 0;
  const roofScore = roofVision.hasRoof ? 0.15 : 0;
  const businessTypeScore = isCommercialType(poi.type) ? 0.15 : 0.05;

  return clamp(
    distanceScore * 0.35 +
      addressScore * 0.2 +
      contactScore +
      roofScore +
      businessTypeScore,
    0,
    1,
  );
}

function calculateLeadScore({
  roofArea,
  buildingType,
  hasVerifiedContact,
  pvStatus,
  visionConfidence,
}) {
  const areaScore = clamp(roofArea / 20, 0, 45);
  const typeScore = getBuildingTypeWeight(buildingType);
  const contactScore = hasVerifiedContact ? 15 : 0;
  const pvScore = pvStatus === "existing_pv" ? -10 : 15;
  const confidenceScore = clamp(visionConfidence * 15, 0, 15);

  return Math.round(clamp(areaScore + typeScore + contactScore + pvScore + confidenceScore, 0, 100));
}

function normalizeRoofVision(value) {
  const result = value || {};
  const confidence = Number(result.confidence || 0);
  const pvCount = Number(result.pvCount || 0);
  const roofCount = Number(result.roofCount || 0);

  return {
    hasRoof: roofCount > 0 || Number(result.roofArea || 0) > 0,
    pvCount,
    roofCount,
    confidence,
    roofArea: Number(result.roofArea || 0),
    detections: Array.isArray(result.detections) ? result.detections : [],
  };
}

function resolvePvStatus(roofVision) {
  if (roofVision.confidence < IDENTIFY_DEFAULTS.minVisionConfidence) {
    return "manual_review";
  }

  return roofVision.pvCount > 0 ? "existing_pv" : "available_roof";
}

function shouldRequireManualReview({ matchedPoi, roofVision }) {
  return (
    roofVision.confidence < IDENTIFY_DEFAULTS.minVisionConfidence ||
    matchedPoi.matchConfidence < IDENTIFY_DEFAULTS.minContactConfidence
  );
}

function buildSatelliteTileRequest({ point, zoom }) {
  return {
    lat: point.lat,
    lng: point.lng,
    zoom,
    layer: "satellite",
  };
}

async function normalizeToGcj02({ point, source, coordinateService }) {
  if (source === "gcj02") {
    return point;
  }

  return coordinateService.convert({
    from: source,
    to: "gcj02",
    point,
  });
}

function normalizePlace(value) {
  return {
    name: value?.name || "",
    address: value?.address || "",
    type: value?.type || "",
  };
}

function normalizePoiCandidates(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((poi) => ({
    id: String(poi.id || ""),
    name: String(poi.name || ""),
    address: String(poi.address || ""),
    type: String(poi.type || ""),
    contact: normalizeContact(poi.contact),
    location: poi.location || null,
  }));
}

function normalizeBuildingType(type) {
  const text = String(type || "");
  if (/factory|industrial|园区|厂房|仓储|工业/i.test(text)) return "industrial";
  if (/shop|mall|store|商业|商铺|市场/i.test(text)) return "commercial";
  if (/house|residential|住宅|居民|自建房/i.test(text)) return "residential";
  return "unknown";
}

function getBuildingTypeWeight(type) {
  const weights = {
    industrial: 25,
    commercial: 18,
    residential: 10,
    unknown: 6,
  };

  return weights[type] || weights.unknown;
}

function estimateRoofAreaByType(type) {
  const fallbackArea = {
    industrial: 1200,
    commercial: 280,
    residential: 120,
    unknown: 100,
  };

  return fallbackArea[type] || fallbackArea.unknown;
}

function isCommercialType(type) {
  return ["industrial", "commercial"].includes(normalizeBuildingType(type));
}

function scoreDistance(poiLocation, coordinate) {
  if (!poiLocation) {
    return 0.3;
  }

  const distanceMeters = approximateDistanceMeters(poiLocation, coordinate);
  return clamp(1 - distanceMeters / IDENTIFY_DEFAULTS.nearbyRadiusMeters, 0, 1);
}

function scoreTextOverlap(a = "", b = "") {
  if (!a || !b) {
    return 0;
  }

  const left = new Set(String(a).replace(/\s/g, "").split(""));
  const right = new Set(String(b).replace(/\s/g, "").split(""));
  const intersection = [...left].filter((char) => right.has(char)).length;
  const union = new Set([...left, ...right]).size || 1;

  return intersection / union;
}

function approximateDistanceMeters(a, b) {
  const lat1 = Number(a.lat);
  const lng1 = Number(a.lng);
  const lat2 = Number(b.lat);
  const lng2 = Number(b.lng);

  if ([lat1, lng1, lat2, lng2].some(Number.isNaN)) {
    return IDENTIFY_DEFAULTS.nearbyRadiusMeters;
  }

  const earthRadius = 6371000;
  const toRad = (degree) => (degree * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;

  return earthRadius * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

function normalizeContact(contact) {
  return String(contact || "")
    .split(";")[0]
    .replace(/[^\d+\-]/g, "")
    .slice(0, 24);
}

function maskContact(contact) {
  const normalized = normalizeContact(contact);
  if (normalized.length < 7) {
    return "";
  }

  return `${normalized.slice(0, 3)}****${normalized.slice(-4)}`;
}

function summarizeAddressLevel(address) {
  const text = String(address || "").trim();
  if (!text) {
    return "Unknown Area";
  }

  return text.split(/[路街道区县市省]/).slice(0, 2).join("") || "Masked Area";
}

function buildEvidenceSource(results) {
  return {
    geocode: results.geoResult.status,
    poi: results.poiResult.status,
    vision: results.visionResult.status,
  };
}

function readSettledValue(result) {
  return result.status === "fulfilled" ? result.value : null;
}

function assertAdapters(adapters) {
  if (!adapters?.mapService || !adapters?.visionService || !adapters?.coordinateService) {
    throw new Error("identifyMapClick requires map, vision and coordinate service adapters.");
  }
}

function assertValidCoordinate({ lat, lng }) {
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    throw new Error("Invalid coordinate.");
  }

  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    throw new Error("Coordinate out of range.");
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}
