document.getElementById('toggle_code').addEventListener('click', code_toggle);
var loader = document.getElementById('loader');

function code_toggle(){
  var pres = document.querySelectorAll('pre');
  for (var i = 0; i < pres.length; i++) {
    pres[i].style.display = pres[i].style.display == 'none' ? 'block' : 'none';
  }
}

code_toggle()

function redraw(){
  var figs = document.getElementsByClassName("js-plotly-plot")
  for (var i = 0; i < figs.length; i++) {
    console.log("redrawing fig: " + i)
    Plotly.redraw(figs[i])
  }
}

console.log("Start");
setTimeout(function(){
    console.log("Redraw");
    redraw();
    loader.style.display="none";
}, 2000);
console.log("End");