/* Restaurant inspection risk: dashboard.
   Reads data/dashboard.json (written by scripts/build_dashboard_data.py from results/) and draws everything
   with plain DOM and inline SVG. Every figure shown is computed here from that file; none is typed into the
   page. Text from the data is always inserted with textContent. */
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var DATA_URL = 'data/dashboard.json';

  var META = {
    gbm: { label: 'Boosted model', short: 'Boosted', cls: 'c-blue', priority: 0 },
    previous_result_rule: { label: 'Previous-result rule', short: 'Rule', cls: 'c-orange', priority: 3 },
    logistic: { label: 'Logistic regression', short: 'Logistic', cls: 'c-aqua', priority: 4 },
    base_rate: { label: 'No model (base rate)', short: 'No model', cls: 'c-ref', priority: 2 },
    perfect_ranking: { label: 'Best possible ranking', short: 'Best possible', cls: 'c-best', priority: 1 }
  };
  var DRAW_ORDER = ['perfect_ranking', 'base_rate', 'previous_result_rule', 'logistic', 'gbm'];
  var LEGEND_ORDER = ['gbm', 'previous_result_rule', 'logistic', 'base_rate', 'perfect_ranking'];
  var SPLIT_NAMES = { valid: 'validation', test: 'test', post: 'after the test' };

  /* ---------- small helpers ---------- */

  function appendKids(node, kids) {
    kids.forEach(function (kid) {
      if (Array.isArray(kid)) { appendKids(node, kid); return; }
      if (kid === null || kid === undefined || kid === false) return;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    });
  }

  function make(create, attrs) {
    var node = create();
    Object.keys(attrs || {}).forEach(function (key) {
      var value = attrs[key];
      if (value === null || value === undefined || value === false) return;
      if (key === 'text') node.textContent = value;
      else node.setAttribute(key, value === true ? '' : value);
    });
    appendKids(node, Array.prototype.slice.call(arguments, 2));
    return node;
  }
  function h(tag, attrs) {
    var args = [function () { return document.createElement(tag); }, attrs].concat(Array.prototype.slice.call(arguments, 2));
    return make.apply(null, args);
  }
  function s(tag, attrs) {
    var args = [function () { return document.createElementNS(NS, tag); }, attrs].concat(Array.prototype.slice.call(arguments, 2));
    return make.apply(null, args);
  }

  var fmt = {
    pct: function (x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + '%'; },
    dec: function (x, d) { return x.toFixed(d === undefined ? 3 : d); },
    lift: function (x) { return x.toFixed(2) + '×'; },
    n1: function (x) { return x.toFixed(1); },
    int: function (x) { return Math.round(x).toLocaleString('en-US'); }
  };

  function linear(d0, d1, r0, r1) {
    return function (v) { return r0 + (v - d0) * (r1 - r0) / (d1 - d0); };
  }
  function range(lo, hi, step) {
    var out = [];
    for (var v = lo; v <= hi + step / 1000; v += step) out.push(Math.round(v * 10000) / 10000);
    return out;
  }
  function niceRange(min, max, step) {
    return { lo: Math.floor(min / step + 1e-9) * step, hi: Math.ceil(max / step - 1e-9) * step };
  }
  function niceMax(v) {
    var p = Math.pow(10, Math.floor(Math.log10(v)));
    var f = v / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function monthShort(ym) { return MONTHS[Number(ym.slice(5, 7)) - 1]; }
  function monthLong(ym) { return monthShort(ym) + ' ' + ym.slice(0, 4); }
  function dayLong(iso) {
    var d = new Date(iso + 'T00:00:00Z');
    return d.toLocaleDateString('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric', year: 'numeric' });
  }
  function dayShort(iso) {
    var d = new Date(iso + 'T00:00:00Z');
    return d.toLocaleDateString('en-US', { timeZone: 'UTC', month: 'short', day: 'numeric' });
  }
  function sentence(text) { return text.charAt(0).toUpperCase() + text.slice(1); }

  /* ---------- tooltip (one element, text only) ---------- */

  var tipEl = document.getElementById('tip');
  var tooltip = {
    show: function (title, rows, x, y) {
      tipEl.textContent = '';
      if (title) tipEl.append(h('div', { class: 'tip-title', text: title }));
      rows.forEach(function (row) {
        tipEl.append(h('div', { class: 'tip-row' },
          row.cls ? h('span', { class: 'key ' + row.cls }) : null,
          h('strong', { text: row.value }),
          h('span', { class: 'name', text: row.label })));
      });
      tipEl.hidden = false;
      var w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
      var left = x + 14, top = y + 14;
      if (left + w > window.innerWidth - 8) left = x - w - 14;
      if (left < 8) left = 8;
      if (top + hh > window.innerHeight - 8) top = y - hh - 14;
      if (top < 8) top = 8;
      tipEl.style.left = left + 'px';
      tipEl.style.top = top + 'px';
    },
    hide: function () { tipEl.hidden = true; }
  };

  /* ---------- shared pieces: legend, table twin, card ---------- */

  function legend(items) {
    return h('ul', { class: 'legend' }, items.map(function (item) {
      return h('li', {}, h('span', { class: 'key ' + item.cls + (item.swatch ? ' swatch' : '') }), item.label);
    }));
  }

  function twin(summary, headers, rows) {
    var head = h('tr', {}, headers.map(function (t) { return h('th', { scope: 'col', text: t }); }));
    var body = rows.map(function (row) { return h('tr', {}, row.map(function (cell) { return h('td', { text: cell }); })); });
    return h('details', { class: 'twin' }, h('summary', { text: summary }),
      h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, head), h('tbody', {}, body))));
  }

  function fillCard(id, title, sub, extras) {
    var fig = document.getElementById(id);
    fig.textContent = '';
    var plot = h('div', { class: 'plot' });
    fig.append(h('figcaption', {}, h('h3', { text: title }), h('p', { class: 'sub', text: sub })), plot);
    (extras || []).forEach(function (node) { if (node) fig.append(node); });
    return plot;
  }

  function plotWidth(host, min) { return Math.max(Math.floor(host.clientWidth), min || 260); }

  /* ---------- line chart (x positions numeric; labels supplied) ---------- */

  function lineChart(host, o) {
    host.textContent = '';
    var W = plotWidth(host);
    var H = o.height || 280;
    var showEnds = W >= 440 && o.series.some(function (sr) { return sr.endLabel; });
    var m = { t: o.bands ? 30 : 18, r: showEnds ? 104 : 16, b: o.xTitle ? 52 : 32, l: 48 };
    var x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    var xs = o.xs, n = xs.length;
    var sx = linear(xs[0], xs[n - 1], x0, x1);
    var sy = linear(o.yMin, o.yMax, y0, y1);
    var pxs = xs.map(sx);
    var svg = s('svg', { width: W, height: H, class: 'chart-svg', role: 'img', tabindex: 0, 'aria-label': o.label });
    host.append(svg);

    if (o.bands) {
      o.bands.forEach(function (b) {
        var bx0 = b.from === 0 ? x0 : (pxs[b.from] + pxs[b.from - 1]) / 2;
        var bx1 = b.to === n - 1 ? x1 : (pxs[b.to] + pxs[b.to + 1]) / 2;
        if (b.shade) svg.append(s('rect', { x: bx0, y: y1, width: bx1 - bx0, height: y0 - y1, class: 'band-bg' }));
        svg.append(s('text', { x: (bx0 + bx1) / 2, y: y1 - 10, 'text-anchor': 'middle', class: 'band-label' }, b.label));
      });
    }
    o.yTicks.forEach(function (t) {
      var y = sy(t);
      svg.append(s('line', { x1: x0, x2: x1, y1: y, y2: y, class: t === o.yMin ? 'axis' : 'grid' }),
        s('text', { x: x0 - 8, y: y + 4, 'text-anchor': 'end', class: 'tick' }, o.yFmt(t)));
    });
    o.xTicks.forEach(function (t) {
      var x = sx(t.x);
      svg.append(s('line', { x1: x, x2: x, y1: y0, y2: y0 + 4, class: 'axis' }),
        s('text', { x: x, y: y0 + 18, 'text-anchor': 'middle', class: 'tick' }, t.label));
    });
    if (o.xTitle) svg.append(s('text', { x: (x0 + x1) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, o.xTitle));

    o.series.forEach(function (sr) {
      if (sr.lo) {
        var up = xs.map(function (x, i) { return sx(x) + ',' + sy(sr.hi[i]); });
        var down = xs.map(function (x, i) { return sx(x) + ',' + sy(sr.lo[i]); }).reverse();
        svg.append(s('polygon', { points: up.concat(down).join(' '), class: 'range-band ' + sr.cls }));
      }
      svg.append(s('polyline', { points: xs.map(function (x, i) { return sx(x) + ',' + sy(sr.values[i]); }).join(' '), class: 'line ' + sr.cls }));
    });

    if (o.marker) {
      var mx = sx(o.marker.x);
      var anchor = mx < x0 + 40 ? 'start' : mx > x1 - 40 ? 'end' : 'middle';
      svg.append(s('line', { x1: mx, x2: mx, y1: y1, y2: y0, class: 'marker' }),
        s('text', { x: mx, y: y1 - 6, 'text-anchor': anchor, class: 'marker-label' }, o.marker.label));
      o.series.filter(function (sr) { return sr.markerDot; }).forEach(function (sr) {
        svg.append(s('circle', { cx: mx, cy: sy(sr.values[o.marker.index]), r: 4.5, class: 'dot ' + sr.cls }));
      });
    }

    if (showEnds) {
      var placed = [];
      o.series.filter(function (sr) { return sr.endLabel; })
        .sort(function (a, b) { return a.priority - b.priority; })
        .forEach(function (sr) {
          var y = sy(sr.values[n - 1]);
          if (placed.some(function (p) { return Math.abs(p - y) < 15; })) return;
          placed.push(y);
          svg.append(s('line', { x1: x1 + 6, x2: x1 + 20, y1: y, y2: y, class: 'line ' + sr.cls }),
            s('text', { x: x1 + 26, y: y + 4, class: 'end-label' }, sr.endLabel));
        });
    }

    var hover = s('g', { style: 'visibility:hidden' });
    var cross = s('line', { y1: y1, y2: y0, class: 'cross' });
    hover.append(cross);
    var dots = o.series.filter(function (sr) { return !sr.noHover; }).map(function (sr) {
      var c = s('circle', { r: 4.5, class: 'dot ' + sr.cls });
      hover.append(c);
      return { sr: sr, c: c };
    });
    svg.append(hover);

    var current = o.startIndex || 0;
    function show(i, cx, cy) {
      current = i;
      cross.setAttribute('x1', pxs[i]);
      cross.setAttribute('x2', pxs[i]);
      dots.forEach(function (d) { d.c.setAttribute('cx', pxs[i]); d.c.setAttribute('cy', sy(d.sr.values[i])); });
      hover.style.visibility = 'visible';
      var rows = o.series.filter(function (sr) { return !sr.noTip; })
        .slice().sort(function (a, b) { return b.values[i] - a.values[i]; })
        .map(function (sr) { return { cls: sr.cls, label: sr.label, value: o.valueFmt(sr.values[i]) }; });
      if (o.tipExtra) rows = rows.concat(o.tipExtra(i));
      tooltip.show(o.tipTitle(i), rows, cx, cy);
    }
    function hide() { hover.style.visibility = 'hidden'; tooltip.hide(); }
    function anchorPoint(i) {
      var r = svg.getBoundingClientRect();
      return [r.left + pxs[i], r.top + (y0 + y1) / 2];
    }
    svg.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect();
      var px = e.clientX - r.left;
      if (px < x0 - 12 || px > x1 + 12) { hide(); return; }
      var best = 0;
      for (var i = 1; i < n; i++) if (Math.abs(pxs[i] - px) < Math.abs(pxs[best] - px)) best = i;
      show(best, e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () { var a = anchorPoint(current); show(current, a[0], a[1]); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (e) {
      var i = current;
      if (e.key === 'ArrowLeft') i = Math.max(0, i - 1);
      else if (e.key === 'ArrowRight') i = Math.min(n - 1, i + 1);
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = n - 1;
      else return;
      e.preventDefault();
      var a = anchorPoint(i);
      show(i, a[0], a[1]);
    });
  }

  /* ---------- reliability diagram ---------- */

  function reliabilityChart(host, o) {
    host.textContent = '';
    var W = plotWidth(host);
    var H = Math.min(Math.round(W * 0.85), 400);
    var m = { t: 16, r: 16, b: 52, l: 56 };
    var x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    var sx = linear(0, 1, x0, x1), sy = linear(0, 1, y0, y1);
    var svg = s('svg', { width: W, height: H, class: 'chart-svg', role: 'img', tabindex: 0, 'aria-label': o.label });
    host.append(svg);
    range(0, 1, 0.2).forEach(function (t) {
      svg.append(s('line', { x1: x0, x2: x1, y1: sy(t), y2: sy(t), class: t === 0 ? 'axis' : 'grid' }),
        s('text', { x: x0 - 8, y: sy(t) + 4, 'text-anchor': 'end', class: 'tick' }, fmt.pct(t, 0)),
        s('text', { x: sx(t), y: y0 + 18, 'text-anchor': 'middle', class: 'tick' }, fmt.pct(t, 0)));
    });
    svg.append(s('text', { x: (x0 + x1) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-title' }, 'Mean predicted probability in the bin'));
    svg.append(s('text', { x: 14, y: (y0 + y1) / 2, 'text-anchor': 'middle', class: 'axis-title', transform: 'rotate(-90 14 ' + ((y0 + y1) / 2) + ')' }, 'Observed share'));
    svg.append(s('line', { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), class: 'diagonal' }));

    var points = [];
    o.series.forEach(function (sr) {
      var pts = sr.points.filter(function (p) { return p.n > 0 && p.x !== null; });
      svg.append(s('polyline', { points: pts.map(function (p) { return sx(p.x) + ',' + sy(p.y); }).join(' '), class: 'line ' + sr.cls }));
      pts.forEach(function (p) { points.push({ sr: sr, p: p, px: sx(p.x), py: sy(p.y) }); });
    });
    var circles = points.map(function (pt) {
      var c = s('circle', { cx: pt.px, cy: pt.py, r: 5, class: 'dot ' + pt.sr.cls });
      svg.append(c);
      return c;
    });
    var current = 0;
    function show(i, cx, cy) {
      current = i;
      circles.forEach(function (c, k) { c.setAttribute('class', 'dot ' + points[k].sr.cls + (k === i ? ' hot' : '')); });
      var pt = points[i];
      tooltip.show(pt.sr.label + ', bin ' + fmt.pct(pt.p.lo, 0) + ' to ' + fmt.pct(pt.p.hi, 0), [
        { cls: pt.sr.cls, label: 'observed share', value: fmt.pct(pt.p.y) },
        { label: 'mean predicted', value: fmt.pct(pt.p.x) },
        { label: 'inspections in the bin', value: fmt.int(pt.p.n) }
      ], cx, cy);
    }
    function hide() { circles.forEach(function (c, k) { c.setAttribute('class', 'dot ' + points[k].sr.cls); }); tooltip.hide(); }
    svg.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect();
      var px = e.clientX - r.left, py = e.clientY - r.top;
      var best = -1, bestD = 28 * 28;
      points.forEach(function (pt, i) {
        var d = (pt.px - px) * (pt.px - px) + (pt.py - py) * (pt.py - py);
        if (d < bestD) { bestD = d; best = i; }
      });
      if (best < 0) hide(); else show(best, e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', hide);
    function anchor(i) { var r = svg.getBoundingClientRect(); return [r.left + points[i].px, r.top + points[i].py]; }
    svg.addEventListener('focus', function () { if (points.length) { var a = anchor(current); show(current, a[0], a[1]); } });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function (e) {
      var i = current;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') i = Math.max(0, i - 1);
      else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') i = Math.min(points.length - 1, i + 1);
      else return;
      e.preventDefault();
      var a = anchor(i);
      show(i, a[0], a[1]);
    });
  }

  /* ---------- horizontal bars with a range whisker ---------- */

  function barPath(x, y, w, hgt, r) {
    var rr = Math.max(0, Math.min(r, w, hgt / 2));
    return 'M' + x + ',' + y + 'h' + (w - rr) + 'a' + rr + ',' + rr + ' 0 0 1 ' + rr + ',' + rr +
      'v' + (hgt - 2 * rr) + 'a' + rr + ',' + rr + ' 0 0 1 ' + (-rr) + ',' + rr + 'h' + (-(w - rr)) + 'z';
  }

  function barChart(host, o) {
    host.textContent = '';
    var W = plotWidth(host, 240);
    var rowH = 54;
    var m = { t: 4, r: 70, b: 28, l: 4 };
    var H = m.t + o.rows.length * rowH + m.b;
    var x0 = m.l, x1 = W - m.r;
    var sx = linear(0, o.max, x0, x1);
    var svg = s('svg', { width: W, height: H, class: 'chart-svg', role: 'img', 'aria-label': o.label });
    host.append(svg);
    var yAxisBottom = H - m.b;
    o.ticks.forEach(function (t) {
      svg.append(s('line', { x1: sx(t), x2: sx(t), y1: m.t, y2: yAxisBottom, class: t === 0 ? 'axis' : 'grid' }),
        s('text', { x: sx(t), y: yAxisBottom + 18, 'text-anchor': 'middle', class: 'tick' }, o.tickFmt(t)));
    });
    o.rows.forEach(function (row, i) {
      var y = m.t + i * rowH;
      var g = s('g', {});
      g.append(s('text', { x: x0, y: y + 14, class: 'bar-label' }, row.label));
      var bw = Math.max(0, sx(row.value) - x0);
      g.append(s('path', { d: barPath(x0, y + 22, bw, 16, 4), class: 'bar c-blue' }));
      if (row.lo !== undefined && row.hi !== undefined) {
        var cy = y + 30;
        g.append(s('line', { x1: sx(row.lo), x2: sx(row.hi), y1: cy, y2: cy, class: 'whisker' }),
          s('line', { x1: sx(row.lo), x2: sx(row.lo), y1: cy - 4, y2: cy + 4, class: 'whisker' }),
          s('line', { x1: sx(row.hi), x2: sx(row.hi), y1: cy - 4, y2: cy + 4, class: 'whisker' }));
      }
      var labelX = Math.max(sx(row.value), row.hi !== undefined ? sx(row.hi) : 0) + 10;
      g.append(s('text', { x: labelX, y: y + 34, class: 'value-label' }, o.fmt(row.value)));
      var hit = s('rect', { x: 0, y: y, width: W, height: rowH - 4, class: 'hit' });
      hit.addEventListener('pointermove', function (e) { tooltip.show(row.label, row.tip, e.clientX, e.clientY); });
      hit.addEventListener('pointerleave', tooltip.hide);
      g.append(hit);
      svg.append(g);
    });
  }

  /* ---------- drift heat table ---------- */

  var STATUS_TEXT = { stable: 'stable', moderate: 'moderate shift', major: 'major shift', none: 'no value' };

  function featureName(name) {
    var known = {
      prev_ci_n_crit: 'Previous initial: critical violations',
      prev_ci_n_viol: 'Previous initial: violations',
      prev_ci_score: 'Previous initial: score',
      prev_ci_closed: 'Previous initial: closed',
      days_since_prev_ci: 'Days since previous initial',
      last_n_crit: 'Latest earlier: critical violations',
      last_score: 'Latest earlier: score',
      days_since_last_inspection: 'Days since latest earlier inspection',
      n_inspection_days_365: 'Inspection days in the last year',
      ever_closed_prior: 'Ever closed before',
      zip_prior_rate: 'ZIP code: earlier hit rate',
      cuisine_prior_rate: 'Cuisine: earlier hit rate',
      month: 'Calendar month',
      weekday: 'Weekday',
      boro: 'Borough',
      cuisine_group: 'Cuisine group',
      __prediction__: 'Prediction (score)'
    };
    if (known[name]) return known[name];
    var a = /^prev_ci_fam_(\w+)$/.exec(name);
    if (a) return 'Previous initial cited family ' + a[1];
    var b = /^fam_(\w+)_prior_share$/.exec(name);
    if (b) return 'Family ' + b[1] + ': share of earlier days';
    return name;
  }

  function heatTable(host, d) {
    host.textContent = '';
    var cut = d.cutoffs;
    function status(v) { return v === null ? 'none' : v >= cut.major ? 'major' : v >= cut.moderate ? 'moderate' : 'stable'; }

    var groups = [];
    d.splits.forEach(function (sp, i) {
      var last = groups[groups.length - 1];
      if (last && last.split === sp) last.span++; else groups.push({ split: sp, span: 1, start: i });
    });
    var head1 = h('tr', {}, h('th', { scope: 'col', text: 'Feature' }));
    groups.forEach(function (g) { head1.append(h('th', { scope: 'colgroup', colspan: g.span, class: 'group-head', text: SPLIT_NAMES[g.split] || g.split })); });
    var head2 = h('tr', {}, h('th', { scope: 'col' }, h('span', { class: 'group-head', text: 'Month' })));
    d.months.forEach(function (mth) { head2.append(h('th', { scope: 'col', text: monthShort(mth) + ' ' + mth.slice(2, 4) })); });

    var body = h('tbody', {});
    function addRow(name, values, divider) {
      var tr = h('tr', { class: divider ? 'divider' : null }, h('th', { scope: 'row', text: featureName(name) }));
      values.forEach(function (v, c) {
        var st = status(v);
        var td = h('td', { class: st, text: v === null ? '–' : v.toFixed(2) });
        function showCell(e) {
          tooltip.show(featureName(name) + ', ' + monthLong(d.months[c]), [
            { label: STATUS_TEXT[st] + (name === '__prediction__' && v === null ? ' (reference window)' : ''), value: v === null ? 'no PSI' : 'PSI ' + v.toFixed(3) }
          ], e.clientX, e.clientY);
        }
        td.addEventListener('pointerenter', showCell);
        td.addEventListener('pointermove', showCell);
        td.addEventListener('pointerleave', tooltip.hide);
        tr.append(td);
      });
      body.append(tr);
    }
    addRow('__prediction__', d.prediction_psi, true);
    d.features.forEach(function (name, r) { addRow(name, d.psi[r], false); });

    host.append(h('div', { class: 'heat-wrap', tabindex: 0, role: 'region', 'aria-label': 'Drift table, scrollable' },
      h('table', { class: 'heat' }, h('thead', {}, head1, head2), body)));
  }

  /* ---------- state ---------- */

  var state = { data: null, capacity: 100, wired: false, request: 0, width: 0, drawers: [], capacityDrawers: [] };

  var root = document.documentElement;
  var main = document.getElementById('main');
  var statusEl = document.getElementById('status');
  var appEl = document.getElementById('app');

  function setStatus(text, error, retry) {
    statusEl.textContent = '';
    statusEl.className = error ? 'status error' : 'status';
    statusEl.hidden = false;
    if (!error) { statusEl.textContent = text; return; }
    statusEl.append(h('p', { text: text }));
    if (location.protocol === 'file:') {
      statusEl.append(h('p', {}, 'This page was opened from a file, and browsers block reading the data file that way. Serve the docs folder instead: ',
        h('code', { text: 'python -m http.server --directory docs' }), ' and open the address it prints.'));
    }
    if (retry) {
      var button = h('button', { type: 'button', class: 'ghost', text: 'Try again' });
      button.addEventListener('click', retry);
      statusEl.append(button);
    }
  }

  function check(data) {
    ['about', 'capacity', 'headline', 'curve', 'weekly', 'calibration', 'split_comparison', 'drift', 'importance', 'model']
      .forEach(function (key) { if (!data[key]) throw new Error('The data file has no "' + key + '" section.'); });
    if (!data.curve.capacities || data.curve.capacities.length === 0) throw new Error('The data file has no capacity results to show.');
    if (!data.weekly.length) throw new Error('The data file has no weekly results to show.');
  }

  /* ---------- tiles ---------- */

  function renderTiles(data) {
    var idx = data.curve.capacities.indexOf(state.capacity);
    var models = data.curve.models;
    var seeds = data.model.seeds.length;
    var host = document.getElementById('tiles');
    host.textContent = '';
    var specs = [
      { metric: 'precision', title: 'Precision', what: 'Of the inspections put first, the share that found the problem.', f: fmt.pct },
      { metric: 'recall', title: 'Recall', what: 'Of all the problem inspections that week, the share that were put first.', f: fmt.pct },
      { metric: 'lift', title: 'Lift', what: 'Precision divided by the share of problem inspections in the whole pool.', f: fmt.lift },
      { metric: 'hits_per_week', title: 'Found per week', what: 'Problem inspections among those put first, on average per week.', f: fmt.n1 }
    ];
    specs.forEach(function (spec) {
      var gbm = models.gbm;
      var list = h('ul', {});
      ['previous_result_rule', 'logistic', 'base_rate'].forEach(function (key) {
        list.append(h('li', {}, h('span', { class: 'key ' + META[key].cls }), META[key].label,
          h('span', { class: 'num', text: spec.f(models[key][spec.metric][idx]) })));
      });
      host.append(h('article', { class: 'tile' },
        h('h3', { text: spec.title + ', ' + META.gbm.label.toLowerCase() }),
        h('p', { class: 'value', text: spec.f(gbm[spec.metric][idx]) }),
        h('p', { class: 'range', text: spec.f(gbm[spec.metric + '_min'][idx]) + ' to ' + spec.f(gbm[spec.metric + '_max'][idx]) + ' across ' + seeds + ' seeds' }),
        h('p', { class: 'what', text: spec.what }),
        list));
    });
  }

  function renderHint(data) {
    var pool = data.capacity.mean_pool_per_week_test;
    var cap = state.capacity;
    var text = cap >= pool
      ? 'That is at least the ' + fmt.int(pool) + ' initial inspections an average test week pools, so almost everything would be visited and ranking matters little.'
      : 'That is about ' + fmt.pct(cap / pool, 0) + ' of the ' + fmt.int(pool) + ' initial inspections an average test week pools.';
    text += ' The default of ' + data.capacity.default + ' a week is my assumption; the real schedule is not in the data.';
    document.getElementById('capacity-hint').textContent = text;
    document.getElementById('capacity-out').textContent = String(cap);
  }

  /* ---------- the cards ---------- */

  function capacitySeries(data, metric) {
    var models = data.curve.models;
    return DRAW_ORDER.map(function (key) {
      var sr = {
        id: key, label: META[key].label, cls: META[key].cls, endLabel: META[key].short, priority: META[key].priority,
        values: models[key][metric], markerDot: key === 'gbm' || key === 'previous_result_rule' || key === 'logistic'
      };
      if (key === 'gbm') { sr.lo = models[key][metric + '_min']; sr.hi = models[key][metric + '_max']; }
      return sr;
    });
  }

  function buildCapacityCard(data, id, metric, title, sub) {
    var caps = data.curve.capacities;
    var series = capacitySeries(data, metric);
    var plot = fillCard(id, title, sub, [
      legend(LEGEND_ORDER.map(function (k) { return { cls: META[k].cls, label: META[k].label }; })),
      twin('View as a table', ['Inspections per week'].concat(LEGEND_ORDER.map(function (k) { return META[k].label; })),
        caps.map(function (c, i) { return [String(c)].concat(LEGEND_ORDER.map(function (k) { return fmt.pct(data.curve.models[k][metric][i]); })); }))
    ]);
    function draw() {
      var idx = caps.indexOf(state.capacity);
      lineChart(plot, {
        label: title + ' by number of inspections put first per week. Use the arrow keys to read values.',
        xs: caps, series: series, startIndex: idx,
        yMin: 0, yMax: 1, yTicks: range(0, 1, 0.2), yFmt: function (v) { return fmt.pct(v, 0); },
        xTicks: caps.filter(function (c, i) { return i === 0 || c % 50 === 0; }).map(function (c) { return { x: c, label: String(c) }; }),
        xTitle: 'Inspections put first per week',
        marker: { x: state.capacity, index: idx, label: state.capacity + ' a week' },
        valueFmt: function (v) { return fmt.pct(v); },
        tipTitle: function (i) { return caps[i] + ' put first per week'; }
      });
    }
    state.capacityDrawers.push(draw);
  }

  function buildWeekly(data) {
    var weeks = data.weekly;
    var xs = weeks.map(function (w, i) { return i; });
    var byModel = {
      gbm: weeks.map(function (w) { return w.hits_gbm / w.visits; }),
      previous_result_rule: weeks.map(function (w) { return w.hits_previous_result_rule / w.visits; }),
      logistic: weeks.map(function (w) { return w.hits_logistic / w.visits; }),
      base_rate: weeks.map(function (w) { return w.positives / w.pool; })
    };
    var ahead = function (a, b) { return a.filter(function (v, i) { return v > b[i]; }).length; };
    var cap = data.capacity.default;
    var sub = 'Precision of the top ' + cap + ' of each test week. The boosted model beat that week’s base rate in ' +
      ahead(byModel.gbm, byModel.base_rate) + ' of ' + weeks.length + ' weeks and the previous-result rule in ' +
      ahead(byModel.gbm, byModel.previous_result_rule) + '. This chart is fixed at ' + cap + ' a week and does not follow the slider.';
    var keys = ['base_rate', 'previous_result_rule', 'logistic', 'gbm'];
    var series = keys.map(function (k) {
      return { id: k, label: k === 'base_rate' ? 'Base rate of the week’s pool' : META[k].label, cls: META[k].cls, values: byModel[k], endLabel: k === 'base_rate' ? 'Base rate' : META[k].short, priority: META[k].priority };
    });
    var all = [].concat.apply([], keys.map(function (k) { return byModel[k]; }));
    var r = niceRange(Math.min.apply(null, all), Math.max.apply(null, all), 0.1);
    var seen = {};
    var xTicks = [];
    weeks.forEach(function (w, i) {
      var key = w.week.slice(0, 7);
      if (seen[key]) return;
      seen[key] = true;
      xTicks.push({ x: i, label: monthShort(key) });
    });
    var plot = fillCard('card-weekly', 'Precision in each of the ' + weeks.length + ' test weeks', sub, [
      legend([{ cls: META.gbm.cls, label: META.gbm.label }, { cls: META.previous_result_rule.cls, label: META.previous_result_rule.label },
        { cls: META.logistic.cls, label: META.logistic.label }, { cls: 'c-ref', label: 'Base rate of the week’s pool' }]),
      twin('View as a table', ['Week starting', 'Pool', 'Problem inspections in pool', 'Put first', 'Boosted: found', 'Rule: found', 'Logistic: found'],
        weeks.map(function (w) { return [w.week, String(w.pool), String(w.positives), String(w.visits), String(w.hits_gbm), String(w.hits_previous_result_rule), String(w.hits_logistic)]; }))
    ]);
    function draw() {
      lineChart(plot, {
        label: 'Precision of the top ' + cap + ' per week, by test week. Use the arrow keys to read values.',
        xs: xs, series: series, height: 300,
        yMin: r.lo, yMax: r.hi, yTicks: range(r.lo, r.hi, 0.1), yFmt: function (v) { return fmt.pct(v, 0); },
        xTicks: xTicks, valueFmt: function (v) { return fmt.pct(v); },
        tipTitle: function (i) { return 'Week of ' + dayLong(weeks[i].week); },
        tipExtra: function (i) { return [{ label: 'inspections in the pool', value: String(weeks[i].pool) }, { label: 'problem inspections in the pool', value: String(weeks[i].positives) }]; }
      });
    }
    state.drawers.push(draw);
  }

  function buildCalibration(data) {
    var cal = data.calibration;
    var method = { platt: 'Platt', isotonic: 'Isotonic' }[cal.method] || cal.method;
    var windows = data.about.windows;
    var seeds = data.model.seeds.length;
    var tiles = document.getElementById('calibration-tiles');
    tiles.textContent = '';
    [['Expected calibration error, raw scores', fmt.dec(cal.ece_before, 3)],
      ['Expected calibration error, after ' + method + ' scaling', fmt.dec(cal.ece_after, 3)],
      ['Brier score, raw scores', fmt.dec(cal.brier_before, 4)],
      ['Brier score, after ' + method + ' scaling', fmt.dec(cal.brier_after, 4)]].forEach(function (t) {
      tiles.append(h('article', { class: 'tile' }, h('h3', { text: t[0] }), h('p', { class: 'value', text: t[1] })));
    });
    var worse = cal.ece_after > cal.ece_before;
    document.getElementById('calibration-note').textContent =
      'On the test window the ' + method + ' calibrator, fitted on the validation window, ' + (worse ? 'made calibration worse' : 'improved calibration') +
      ' (error ' + fmt.dec(cal.ece_before, 3) + ' to ' + fmt.dec(cal.ece_after, 3) + '; lower is better). The share of inspections with a hit was ' +
      fmt.pct(windows.validation.positive_share) + ' in the validation window and ' + fmt.pct(windows.test.positive_share) +
      ' in the test window, and a calibrator fitted on one cannot know about the other. The Brier score went from ' + fmt.dec(cal.brier_before, 4) + ' to ' + fmt.dec(cal.brier_after, 4) + ' (also lower is better). Scores are the average of the ' + seeds + ' seed models.';

    var rel = fillCard('card-reliability', 'Reliability on the test window',
      'Each point is a bin of predictions. On the diagonal, the predicted probability equals the observed share. Hover or use the arrow keys for the bin counts.', [
        legend([{ cls: 'c-blue', label: 'Raw scores' }, { cls: 'c-orange', label: 'After ' + method + ' scaling' }, { cls: 'c-best', label: 'Perfect calibration' }]),
        twin('View as a table', ['Bin', 'Raw: inspections', 'Raw: mean predicted', 'Raw: observed', 'Scaled: inspections', 'Scaled: mean predicted', 'Scaled: observed'],
          cal.reliability_before.map(function (b, i) {
            var a = cal.reliability_after[i];
            function f(v) { return v === null ? 'none' : fmt.pct(v); }
            return [fmt.pct(b.lo, 0) + ' to ' + fmt.pct(b.hi, 0), String(b.n), f(b.mean_predicted), f(b.observed_rate), String(a.n), f(a.mean_predicted), f(a.observed_rate)];
          }))
      ]);
    var seriesRel = [
      { cls: 'c-blue', label: 'Raw scores', points: cal.reliability_before.map(function (b) { return { x: b.mean_predicted, y: b.observed_rate, n: b.n, lo: b.lo, hi: b.hi }; }) },
      { cls: 'c-orange', label: 'After ' + method + ' scaling', points: cal.reliability_after.map(function (b) { return { x: b.mean_predicted, y: b.observed_rate, n: b.n, lo: b.lo, hi: b.hi }; }) }
    ];
    state.drawers.push(function () {
      reliabilityChart(rel, { series: seriesRel, label: 'Reliability diagram, predicted probability against observed share, raw and scaled.' });
    });

    var months = cal.monthly;
    var xs = months.map(function (m, i) { return i; });
    var obs = months.map(function (m) { return m.observed_rate; });
    var raw = months.map(function (m) { return m.mean_raw_probability; });
    var scaled = months.map(function (m) { return m.mean_calibrated_probability; });
    var all = obs.concat(raw, scaled);
    var r = niceRange(Math.min.apply(null, all), Math.max.apply(null, all), 0.05);
    var bands = [];
    months.forEach(function (m, i) {
      var last = bands[bands.length - 1];
      if (last && last.split === m.split) last.to = i;
      else bands.push({ split: m.split, from: i, to: i, label: SPLIT_NAMES[m.split] || m.split, shade: bands.length % 2 === 1 });
    });
    var series = [
      { label: 'Observed share with a hit', cls: 'c-ink', values: obs, endLabel: 'Observed', priority: 0 },
      { label: 'Mean raw score', cls: 'c-blue', values: raw, endLabel: 'Raw', priority: 1 },
      { label: 'Mean after ' + method + ' scaling', cls: 'c-orange', values: scaled, endLabel: 'Scaled', priority: 2 }
    ];
    var plot = fillCard('card-monthly', 'Observed share and mean prediction, by month',
      'If the base rate moves, the average prediction should follow it. Months are calendar months of the inspection date.', [
        legend(series.map(function (sr) { return { cls: sr.cls, label: sr.label }; })),
        twin('View as a table', ['Month', 'Window', 'Inspections', 'Observed share', 'Mean raw score', 'Mean scaled score'],
          months.map(function (m) { return [m.month, SPLIT_NAMES[m.split] || m.split, String(m.rows), fmt.pct(m.observed_rate), fmt.pct(m.mean_raw_probability), fmt.pct(m.mean_calibrated_probability)]; }))
      ]);
    var step = Math.max(1, Math.ceil(44 / ((plotWidth(plot) - 64) / months.length)));
    state.drawers.push(function () {
      lineChart(plot, {
        label: 'Observed share with a hit and mean prediction by month. Use the arrow keys to read values.',
        xs: xs, series: series, height: 300, bands: bands,
        yMin: r.lo, yMax: r.hi, yTicks: range(r.lo, r.hi, 0.05), yFmt: function (v) { return fmt.pct(v, 0); },
        xTicks: months.map(function (m, i) { return { x: i, label: monthShort(m.month) + (i === 0 || m.month.slice(5) === '01' ? ' ' + m.month.slice(2, 4) : '') }; })
          .filter(function (t, i) { return i % step === 0; }),
        valueFmt: function (v) { return fmt.pct(v); },
        tipTitle: function (i) { return monthLong(months[i].month); },
        tipExtra: function (i) { return [{ label: 'inspections', value: fmt.int(months[i].rows) }]; }
      });
    });
  }

  function buildSplit(data) {
    var sc = data.split_comparison;
    var seeds = data.model.seeds.length;
    var panels = [
      { metric: 'auc', title: 'Area under the ROC curve', f: function (v) { return fmt.dec(v, 3); } },
      { metric: 'lift', title: 'Lift at the same share of the pool', f: fmt.lift }
    ];
    var fig = document.getElementById('card-split');
    fig.textContent = '';
    fig.append(h('figcaption', {}, h('h3', { text: 'The same model, scored three ways' }),
      h('p', { class: 'sub', text: 'Bars are medians over ' + seeds + ' seeds and the whisker spans the lowest to the highest seed. The time split is the honest one: train on the past, score the future. The random splits mix past and future rows, so a gap between the bars is what a random split would have overstated. Lift uses the top ' + fmt.pct(sc.capacity_fraction, 0) + ' of each pool.' })));
    var wrap = h('div', { class: 'panels' });
    fig.append(wrap);
    panels.forEach(function (p) {
      var host = h('div', { class: 'plot' });
      wrap.append(h('div', { class: 'panel' }, h('h4', { text: p.title }), host));
      var rows = sc.schemes.map(function (sch) {
        var v = sch[p.metric];
        return { label: sch.label, value: v.median, lo: v.min, hi: v.max,
          tip: [{ label: 'median', value: p.f(v.median) }, { label: 'lowest seed', value: p.f(v.min) }, { label: 'highest seed', value: p.f(v.max) }] };
      });
      var max = niceMax(Math.max.apply(null, rows.map(function (r) { return r.hi; })) * 1.05);
      var ticks = range(0, max, max / 4);
      state.drawers.push(function () {
        barChart(host, { rows: rows, max: max, ticks: ticks, tickFmt: function (v) { return p.metric === 'lift' ? v.toFixed(1) : v.toFixed(2); }, fmt: p.f, label: p.title + ' for the time split and two random splits.' });
      });
    });
    fig.append(twin('View as a table', ['Scheme', 'AUC median', 'AUC lowest', 'AUC highest', 'Lift median', 'Lift lowest', 'Lift highest'],
      sc.schemes.map(function (sch) {
        return [sch.label, fmt.dec(sch.auc.median, 4), fmt.dec(sch.auc.min, 4), fmt.dec(sch.auc.max, 4), fmt.lift(sch.lift.median), fmt.lift(sch.lift.min), fmt.lift(sch.lift.max)];
      })));
  }

  function matchedSentence(d) {
    var mh = d.matched_history;
    if (!mh) return 'Whether that alone explains the alerts was not tested.';
    var prev = 'previous initial inspection', hist = 'earlier history';
    return 'To test whether that explains the alerts, I repeated the comparison per window on the inspections whose previous initial inspection was within a year, ' +
      'which sees the same history in every window (' + fmt.pct(mh.windows.test.matched_share, 0) + ' of test-window inspections qualify). ' +
      'On those rows ' + mh.matched.all.major + ' of ' + mh.matched.feature_windows + ' feature-windows are major, against ' + mh.all_rows.all.major + ' of ' + mh.all_rows.feature_windows +
      ' on all rows; for the features about the previous initial inspection the count falls from ' + mh.all_rows.by_group[prev].major + ' of ' + mh.all_rows.by_group[prev].feature_windows +
      ' to ' + mh.matched.by_group[prev].major + '. The features that add up over the whole visible history still alert on matched rows (' + mh.matched.by_group[hist].major + ' of ' +
      mh.matched.by_group[hist].feature_windows + '), because matching does not shorten the past they see.';
  }

  function buildDrift(data) {
    var d = data.drift;
    var windows = data.about.windows;
    var flat = [].concat.apply([], d.psi).filter(function (v) { return v !== null; });
    var counts = { stable: 0, moderate: 0, major: 0 };
    flat.forEach(function (v) { counts[v >= d.cutoffs.major ? 'major' : v >= d.cutoffs.moderate ? 'moderate' : 'stable']++; });
    var predValues = d.prediction_psi.filter(function (v) { return v !== null; });
    var predMax = Math.max.apply(null, predValues);
    var hasMonth = d.features.indexOf('month') >= 0;
    var sub = 'Population stability index (PSI) of each feature, month by month, against the training window. ' +
      counts.major + ' of ' + flat.length + ' feature-months are above the ' + d.cutoffs.major + ' cut-off, ' + counts.moderate + ' are between ' + d.cutoffs.moderate +
      ' and ' + d.cutoffs.major + ', ' + counts.stable + ' are below. For comparison, a random sample of the training window itself, the size of a typical month (' + fmt.int(d.noise_floor.sample_size) +
      ' rows), has a PSI against the whole window of ' + fmt.dec(d.noise_floor.median_over_features_and_seeds, 4) + ' at the median and ' + fmt.dec(d.noise_floor.max_over_features_and_seeds, 3) +
      ' at most, over all features and ' + d.noise_floor.seeds.length + ' seeds, so sampling noise alone does not explain these alerts.';
    var plot = fillCard('card-drift', 'Is the data moving?', sub, [
      legend([{ cls: 'c-stable swatch', label: 'below ' + d.cutoffs.moderate + ', stable' }, { cls: 'c-moderate swatch', label: d.cutoffs.moderate + ' to ' + d.cutoffs.major + ', moderate' }, { cls: 'c-major swatch', label: d.cutoffs.major + ' and above, major' }])
    ]);
    var caption = 'The alerts are mostly in features that describe a restaurant’s earlier inspections. The share of inspections with a visible previous initial inspection is ' +
      fmt.pct(windows.train.previous_initial_visible_share, 0) + ' in the training window, ' + fmt.pct(windows.validation.previous_initial_visible_share, 0) + ' in validation and ' +
      fmt.pct(windows.test.previous_initial_visible_share, 0) + ' in the test window, because the publisher’s history reaches back a fixed three years and the windows move later. ' +
      matchedSentence(d) +
      (hasMonth ? ' The calendar-month row alerts by construction, because one month is compared with a year.' : '') +
      ' The prediction row, which is what the ranking depends on, ' + (predMax < d.cutoffs.moderate
        ? 'stays below the moderate cut-off in every month (highest ' + fmt.dec(predMax, 3) + ').'
        : 'reaches ' + fmt.dec(predMax, 3) + ' in its highest month.') +
      ' Its reference is the validation window, so validation months have no value.';
    plot.parentNode.append(h('p', { class: 'note', text: caption }));
    state.drawers.push(function () { heatTable(plot, d); });
  }

  function buildImportance(data) {
    var rows = data.importance;
    var seeds = data.model.seeds.length;
    var max = niceMax(Math.max.apply(null, rows.map(function (r) { return r.gain_share_max; })) * 1.05);
    var plot = fillCard('card-importance', 'Share of the model’s gain by feature',
      'The ' + rows.length + ' features with the largest share of the total split gain, median over ' + seeds + ' seeds with the lowest to highest seed as the whisker. This says what the model leans on, not why a restaurant has violations. ' +
      'A hit rate is the share of earlier initial inspections that were hits.', [
        twin('View as a table', ['Feature', 'Gain share median', 'Lowest seed', 'Highest seed'],
          rows.map(function (r) { return [featureName(r.feature) + ' (' + r.feature + ')', fmt.pct(r.gain_share_median), fmt.pct(r.gain_share_min), fmt.pct(r.gain_share_max)]; }))
      ]);
    var barRows = rows.map(function (r) {
      return { label: featureName(r.feature), value: r.gain_share_median, lo: r.gain_share_min, hi: r.gain_share_max,
        tip: [{ label: 'median', value: fmt.pct(r.gain_share_median) }, { label: 'lowest seed', value: fmt.pct(r.gain_share_min) }, { label: 'highest seed', value: fmt.pct(r.gain_share_max) }, { label: 'column', value: r.feature }] };
    });
    var ticks = range(0, max, max / 4);
    state.drawers.push(function () {
      barChart(plot, { rows: barRows, max: max, ticks: ticks, tickFmt: function (v) { return fmt.pct(v, 0); }, fmt: function (v) { return fmt.pct(v); }, label: 'Gain share by feature, top ' + rows.length + '.' });
    });
  }

  function buildAbout(data) {
    var a = data.about;
    var w = a.windows;
    var list = document.getElementById('about-list');
    list.textContent = '';
    function item() { list.append(h('li', {}, Array.prototype.slice.call(arguments))); }
    item('Real public data, not synthetic: ', h('a', { href: a.dataset_page, text: a.dataset }), ' from the ' + a.publisher + ', ' + fmt.int(a.citation_rows) +
      ' citation rows, fetched ' + dayLong(a.snapshot_fetched_utc.slice(0, 10)) + '. The publisher last updated the rows on ' + dayLong(a.publisher_rows_updated_utc.slice(0, 10)) + '.');
    item('A hit means ' + a.target + '. The model ranks the initial inspections of each week; it does not say why a restaurant has violations.');
    item('Capacity is an assumption of mine: ' + data.capacity.default + ' a week is ' + fmt.pct(data.capacity.share_of_test_pool, 0) +
      ' of the ' + fmt.int(data.capacity.mean_pool_per_week_test) + ' initial inspections a test week pools on average. The department’s real schedule is not in the data.');
    item('Only restaurants active on the pull date are in the file, and only violations that were sustained or not yet adjudicated. Older months miss restaurants that later closed, and the newest labels are not final. With one snapshot I cannot measure how much they change.');
    item('ZIP code, cuisine and borough are the current snapshot values, not the values on the day of an old inspection.');
    item('The share of inspections with a hit differs by window: ' + fmt.pct(w.train.positive_share) + ' in training, ' + fmt.pct(w.validation.positive_share) + ' in validation, ' +
      fmt.pct(w.test.positive_share) + ' in the test window and ' + fmt.pct(w['after the test'].positive_share) + ' after it.');
    item('No restaurant names, addresses or phone numbers are collected, stored or shown; the page shows aggregates only.');
    item('The windows are ' + Object.keys(w).map(function (k) { return k + ' ' + w[k].first_date + ' to ' + w[k].last_date; }).join('; ') + '.');
    item('Every figure on this page is computed in the browser from data/dashboard.json, which scripts/build_dashboard_data.py writes from the files in results/. ',
      h('a', { href: a.repository, text: 'Repository' }), '.');
    document.getElementById('lede').textContent =
      'Which of the routine inspections due this week should a health department put first? This page replays that question on past weeks, ranking each week’s pool using only what was known before the visit. A hit is ' + a.target + '.';
  }

  /* ---------- draw, resize, wiring ---------- */

  function drawAll() {
    state.drawers.forEach(function (fn) { fn(); });
    state.capacityDrawers.forEach(function (fn) { fn(); });
  }

  function onCapacity() {
    renderHint(state.data);
    renderTiles(state.data);
    state.capacityDrawers.forEach(function (fn) { fn(); });
  }

  var THEMES = ['auto', 'light', 'dark'];
  var THEME_NAMES = { auto: 'Automatic', light: 'Light', dark: 'Dark' };
  var theme = 'auto';
  try { theme = localStorage.getItem('rir-theme') || 'auto'; } catch (e) { theme = 'auto'; }
  if (THEMES.indexOf(theme) < 0) theme = 'auto';
  function applyTheme() {
    if (theme === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', theme);
    var button = document.getElementById('theme');
    if (button) button.textContent = THEME_NAMES[theme] + ' (change)';
  }
  applyTheme();

  function wire() {
    if (state.wired) return;
    state.wired = true;
    var slider = document.getElementById('capacity');
    slider.addEventListener('input', function () { state.capacity = Number(slider.value); onCapacity(); });
    document.getElementById('theme').addEventListener('click', function () {
      theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
      try { localStorage.setItem('rir-theme', theme); } catch (e) { /* storage may be blocked */ }
      applyTheme();
    });
    var frame = 0;
    new ResizeObserver(function () {
      var width = Math.round(appEl.getBoundingClientRect().width);
      if (!state.data || width === state.width) return;
      state.width = width;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(drawAll);
    }).observe(appEl);
    document.addEventListener('scroll', tooltip.hide, { passive: true });
  }

  function build(data) {
    state.data = data;
    state.drawers = [];
    state.capacityDrawers = [];
    var caps = data.curve.capacities;
    var slider = document.getElementById('capacity');
    slider.min = String(caps[0]);
    slider.max = String(caps[caps.length - 1]);
    slider.step = String(caps[1] - caps[0]);
    state.capacity = caps.indexOf(data.capacity.default) >= 0 ? data.capacity.default : caps[0];
    slider.value = String(state.capacity);

    var seeds = data.model.seeds.length;
    var test = data.about.windows.test;
    var testSpan = test.first_date + ' to ' + test.last_date + ', ' + test.weeks + ' weeks';
    document.getElementById('capacity-note').textContent = 'Test window: ' + testSpan + ', ' + fmt.int(test.rows) +
      ' initial inspections pooled. Each line is the median over ' + seeds + ' seeds and the shaded band spans the lowest to the highest seed. ' +
      'Where the logistic line is not visible it sits under the boosted model’s.';
    buildCapacityCard(data, 'card-precision', 'precision', 'Precision as capacity grows',
      'Share of the inspections put first that found the problem, at each capacity.');
    buildCapacityCard(data, 'card-recall', 'recall', 'Recall as capacity grows',
      'Share of the week’s problem inspections that were among those put first, at each capacity.');
    buildWeekly(data);
    buildCalibration(data);
    buildSplit(data);
    buildDrift(data);
    buildImportance(data);
    buildAbout(data);
    wire();

    statusEl.hidden = true;
    appEl.hidden = false;
    state.width = Math.round(appEl.getBoundingClientRect().width);
    renderHint(data);
    renderTiles(data);
    drawAll();
  }

  function start() {
    var mine = ++state.request;
    main.setAttribute('aria-busy', 'true');
    appEl.hidden = true;
    setStatus('Loading results…', false);
    fetch(DATA_URL, { cache: 'no-cache' })
      .then(function (res) {
        if (!res.ok) throw new Error('The data file answered with HTTP ' + res.status + '.');
        return res.json();
      })
      .then(function (data) {
        if (mine !== state.request) return;
        check(data);
        build(data);
        main.setAttribute('aria-busy', 'false');
      })
      .catch(function (err) {
        if (mine !== state.request) return;
        main.setAttribute('aria-busy', 'false');
        setStatus('The results could not be shown. ' + (err && err.message ? err.message : ''), true, start);
      });
  }

  start();
}());
