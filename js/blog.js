/**
 * blog.js — Minimal vanilla JS for blog / experience / growth pages.
 *
 * Features:
 *  · Dark / light theme toggle (persisted in localStorage)
 *  · Reading progress bar
 *  · TOC active-section highlighting (IntersectionObserver)
 *  · Auto-smooth-scroll for TOC links
 *  · Auto-generate TOC from headings  (add data-auto to .toc-nav ul)
 *  · Scroll-triggered fade-in animation
 *  · Experience page section filter
 *  · Mobile nav toggle (same markup as main site)
 */

(function () {
    'use strict';

    /* ------------------------------------------------------------------
     * Theme
     * ---------------------------------------------------------------- */
    var THEME_KEY = 'sk-theme';

    function applyTheme(t) {
        document.body.dataset.theme = t;
        try { localStorage.setItem(THEME_KEY, t); } catch (_) {}
    }

    (function () {
        var saved = '';
        try { saved = localStorage.getItem(THEME_KEY) || ''; } catch (_) {}
        if (!saved) {
            saved = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
        }
        applyTheme(saved);
    }());

    /* ------------------------------------------------------------------
     * DOM-ready
     * ---------------------------------------------------------------- */
    document.addEventListener('DOMContentLoaded', function () {

        /* Theme toggle button */
        var toggle = document.querySelector('.theme-toggle');
        if (toggle) {
            toggle.addEventListener('click', function () {
                applyTheme(document.body.dataset.theme === 'light' ? 'dark' : 'light');
            });
        }

        /* ----------------------------------------------------------
         * Reading progress bar
         * -------------------------------------------------------- */
        var bar = document.querySelector('.reading-progress');
        if (bar) {
            window.addEventListener('scroll', function () {
                var scrolled  = window.scrollY;
                var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
                bar.style.width = maxScroll > 0 ? (scrolled / maxScroll * 100) + '%' : '0%';
            }, { passive: true });
        }

        /* ----------------------------------------------------------
         * Auto-generate TOC
         * -------------------------------------------------------- */
        var autoToc = document.querySelector('.toc-nav ul[data-auto]');
        if (autoToc) {
            var headings = Array.from(
                document.querySelectorAll('.post-body h2, .post-body h3')
            );
            var items = headings.map(function (h) {
                if (!h.id) {
                    h.id = h.textContent
                        .toLowerCase()
                        .trim()
                        .replace(/\s+/g, '-')
                        .replace(/[^a-z0-9-]/g, '');
                }
                var cls = h.tagName === 'H3' ? ' toc-h3' : '';
                return '<li class="' + cls + '">'
                    + '<a href="#' + h.id + '">' + h.textContent + '</a></li>';
            });
            autoToc.innerHTML = items.join('');
        }

        /* ----------------------------------------------------------
         * TOC active highlighting
         * -------------------------------------------------------- */
        var tocLinks = Array.from(document.querySelectorAll('.toc-nav a'));
        if (tocLinks.length && 'IntersectionObserver' in window) {
            var tocHeadings = Array.from(
                document.querySelectorAll('.post-body h2[id], .post-body h3[id], .post-body h4[id]')
            );
            var observer = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        var id = '#' + entry.target.id;
                        tocLinks.forEach(function (a) {
                            a.parentElement.classList.toggle('active', a.getAttribute('href') === id);
                        });
                    }
                });
            }, { rootMargin: '-8% 0px -72% 0px', threshold: 0 });

            tocHeadings.forEach(function (h) { observer.observe(h); });
        }

        /* TOC smooth scroll */
        document.querySelectorAll('.toc-nav a').forEach(function (a) {
            a.addEventListener('click', function (e) {
                var href = this.getAttribute('href');
                if (href && href.charAt(0) === '#') {
                    e.preventDefault();
                    var target = document.getElementById(href.slice(1));
                    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });

        /* ----------------------------------------------------------
         * Scroll-triggered animations (.anim-in elements)
         * -------------------------------------------------------- */
        var animElems = document.querySelectorAll('.anim-in');
        if (animElems.length && 'IntersectionObserver' in window) {
            var animObs = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        entry.target.style.animationPlayState = 'running';
                        animObs.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.08 });

            animElems.forEach(function (el) {
                el.style.animationPlayState = 'paused';
                animObs.observe(el);
            });
        }

        /* ----------------------------------------------------------
         * Experience page section filter
         * -------------------------------------------------------- */
        var filterBtns = document.querySelectorAll('.exp-filter__btn');
        if (filterBtns.length) {
            filterBtns.forEach(function (btn) {
                btn.addEventListener('click', function () {
                    filterBtns.forEach(function (b) { b.classList.remove('active'); });
                    btn.classList.add('active');
                    var filter = btn.dataset.filter;
                    document.querySelectorAll('.exp-section').forEach(function (s) {
                        s.style.display = (filter === 'all' || s.dataset.type === filter) ? '' : 'none';
                    });
                });
            });
        }

        /* ----------------------------------------------------------
         * Mobile nav toggle (replicates main.js behaviour for sub-pages)
         * -------------------------------------------------------- */
        var menuToggle = document.querySelector('.menu-toggle');
        var nav = document.getElementById('main-nav-wrap');
        if (menuToggle && nav) {
            menuToggle.addEventListener('click', function (e) {
                e.preventDefault();
                nav.classList.toggle('open');
            });
        }

    }); // DOMContentLoaded

}());
