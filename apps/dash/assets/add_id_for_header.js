var headers = ["h1", "h2", "h3", "h4"]

for (var i=0, h; h=headers[i]; i++) {  
  var tags = document.getElementsByTagName(h);
  for (var j=0, hi; hi=tags[j]; j++) {
    // remove <> tab
    var tmp = hi.innerHTML.toLowerCase().replace(/<.*?>/g, '')
    // remove !@#$%$%$
    tmp = tmp.replace(/[~`!@#$%^&*(){}\[\];:"'<,.>?\/\\|_+=-]/g, '')
    hi.id = tmp.replace(/\s|\"/g, "-");
  }
}
