console.log("👀regression_coin running !!!");
let anal_btn;
let ele_coin_name;
let timegap;
let res_contain;
let closeBtn;
let loader;
async function get_coinname() {
  ele_coin_name = window.document.getElementById("coinname");
  timegap = document.getElementById("timegap");
  res_contain = document.getElementById("res_contain");
  closeBtn = document.getElementById("closeBtn");
  loader = document.getElementById("loader");
  const conn = await fetch("/coin_name");
  const coinnames = await conn.json();
  let inHtml = "";
  for (let i = 0; i < coinnames.eng_name.length; i++) {
    inHtml += `<option ${i == 0 ? "selected" : ""} value="${
      coinnames.eng_name[i]
    }">
        ${coinnames.han_name[i]}(${coinnames.eng_name[i]})</option>`;
  }
  ele_coin_name.innerHTML = inHtml;
  //console.log(JSON.stringify(coinnames));
  anal_btn = document.getElementById("anal_btn");
  add_Event();
}
function add_Event() {
  closeBtn.addEventListener("click", () => {
    res_contain.style.display = "none";
  });
  anal_btn.addEventListener("click", async function () {
    const coinname = ele_coin_name.value;
    const timegaps = timegap.value;
    const padding = await fetch("/user_data", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        coinname,
        timegaps,
      }),
    }).catch(() => {
      console.log("서버통신오류");
    });
    if (padding) {
      const info_data = await padding.json();
      console.log(info_data);
      let inHtml = "";
      let today_date = new Date();
      today_date.setDate(today_date.getDate() + 1);
      let today_str = today_date.toLocaleString("ko-kr");
      let ghtml = `<h2 style="display:inline;padding:1rem;color:blue">
      ${coinname} 가격 예측 정보</h2>
      <p style="font-size:1.5rem;margin:1rem">최고가오차율(${
        parseInt(info_data["err_rate"]["high"] * 100 * 100) / 100
      }%)&nbsp;&nbsp;&nbsp;&nbsp;
      현재가오차율(${parseInt(info_data["err_rate"]["cur"] * 100 * 100) / 100}%)
      &nbsp;&nbsp;&nbsp;&nbsp;
      최저가오차율(${parseInt(info_data["err_rate"]["low"] * 100 * 100) / 100}%)
      </p>
      <div>
        <h2>훈련 결과 그래프</h2>
        <img style='width:15rem' 
          src="/static/${info_data["graph"][0]}">
        <img style='width:15rem'  
          src="/static/${info_data["graph"][1]}">
      </div>      
      `;

      document.getElementById("anal_data").innerHTML = ghtml;
      for (let data of info_data["ypred"]) {
        //data[0] 현재가
        //data[1] 최고가
        //data[2] 최저가
        inHtml += ` <div style='padding:0.5rem;border:2px solid darkgray;
                                          margin-bottom:1rem'>
            <p style='padding:0.5rem;background:RGB(198, 145, 79)'>
            ${today_str} </p>
            <p style='padding:0.5rem;color:red'>
            최고가 : ${data[1]}</p>
            <p  style='padding:0.5rem'>현재가 : ${data[0]}</p>          
            <p style='padding:0.5rem;color:blue'>최저가 : ${data[2]}</p>\
          </div>
        `;
        today_date.setDate(today_date.getDate() + 1);
        today_str = today_date.toLocaleString("ko-kr");
      }
      loader.style.display = "none";
      document.getElementById("result").innerHTML = inHtml;
      res_contain.style.display = "block";
    }
  });
}
