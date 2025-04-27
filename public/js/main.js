console.log("Page loaded successfully! :)");
const themeSwitch = document.getElementById('themeSwitch');
const savedTheme  = localStorage.getItem('theme') || 'light';
document.documentElement.setAttribute('data-theme', savedTheme);
themeSwitch.checked = (savedTheme === 'dark');
themeSwitch.addEventListener('change', () => {
    console.log('🔘 toggle clicked, checked=', themeSwitch.checked);
    const theme = themeSwitch.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  });