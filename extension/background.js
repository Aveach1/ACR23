function getCurrentURL(){
		chrome.tabs.query({ active: true, currentWindow: true }, function (tabs){
			var tab = tabs[0];
			var testURL = new URL(tab.url)
		})
	}
function redirectLink(){
	getCurrentURL();
	var newLink = ("https://aveach1.gitbub.io?=" + testURL);
	chrome.tabs.create({url: newLink})
}

function isCSPHeader(headerName) {
  return (headerName === 'CONTENT-SECURITY-POLICY') || (headerName === 'X-WEBKIT-CSP');
}
// Listens on new request
chrome.webRequest.onHeadersReceived.addListener((details) => {
  for (let i = 0; i < details.responseHeaders.length; i += 1) {
    if (isCSPHeader(details.responseHeaders[i].name.toUpperCase())) {
      const csp = 'default-src * \'unsafe-inline\' \'unsafe-eval\' data: blob:; ';
      details.responseHeaders[i].value = csp;
    }
  }
  return { // Return the new HTTP header
    responseHeaders: details.responseHeaders,
  };
}, {
  urls: ['<all_urls>'],
  types: ['main_frame'],
}, ['blocking', 'responseHeaders']);
