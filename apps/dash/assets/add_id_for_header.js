var headers = ["h1", "h2", "h3", "h4"]

for (var i=0, h; h=headers[i]; i++) {  
  var tags = document.getElementsByTagName(h);
  for (var j=0, hi; hi=tags[j]; j++) {
    hi.id = hi.innerHTML.toLowerCase().replace(/\s/g, "-");
  }
}


