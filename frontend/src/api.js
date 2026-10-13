async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  const body = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = body?.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  return body;
}

export function getHealth() {
  return request("/api/health");
}

export function getCollections() {
  return request("/api/collections");
}

export function createCollection(name) {
  return request("/api/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function getDocuments(collectionId) {
  return request(`/api/collections/${collectionId}/documents`);
}

export function uploadDocument(collectionId, file) {
  const body = new FormData();
  body.append("file", file);
  return request(`/api/collections/${collectionId}/documents`, {
    method: "POST",
    body,
  });
}

export function deleteDocument(documentId) {
  return request(`/api/documents/${documentId}`, { method: "DELETE" });
}

export function ask(collectionId, payload) {
  return request(`/api/collections/${collectionId}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function getMonitors(collectionId) {
  const query = collectionId ? `?collection_id=${collectionId}` : "";
  return request(`/api/monitors${query}`);
}

export function createMonitor(collectionId, topic) {
  return request("/api/monitors", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection_id: collectionId, topic }),
  });
}

export function runMonitor(monitorId) {
  return request(`/api/monitors/${monitorId}/run`, { method: "POST" });
}
